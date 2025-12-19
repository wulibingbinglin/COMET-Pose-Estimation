import pickle
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch3d.implicitron.tools import  vis_utils
from train_util import *
from train_eval_func import train_or_eval_fn

def train_fn(cfg: DictConfig):
    #######################1. 配置初始化#############################
    OmegaConf.set_struct(cfg, False) # 通过 OmegaConf 关闭了配置的结构性检查，允许动态修改配置文件中的字段。

    accelerator = Accelerator(even_batches=False, device_placement=False, mixed_precision=cfg.mixed_precision)
    # 使用 accelerate 库初始化一个 Accelerator 实例，主要用于分布式训练和混合精度训练的管理。 even_batches=False：这可能表示不对每个设备的批次进行平均 mixed_precision=cfg.mixed_precision：根据配置文件中的设置决定是否启用混合精度训练

    accelerator.print("Model Config:") # 打印出模型的配置。
    accelerator.print(OmegaConf.to_yaml(cfg)) # 以 YAML 格式打印整个配置文件。这个输出有助于调试和查看训练时的所有配置

    accelerator.print(accelerator.state)

##########################设备与调试模式########################
    if cfg.debug: # 检查配置文件中的 debug 字段，决定是否进入调试模式。
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True # 设置 PyTorch 使用确定性的算法，保证每次运行结果一致。常用于调试和重现实验。
        torch.backends.cudnn.benchmark = False # 关闭 cudnn 中的自动调优，适用于输入大小不固定或变动的情况。如果输入数据大小不一致，设置为 False 可以避免性能波动。
    else:
        torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark # 如果不是调试模式，则设置 cudnn 的 benchmark 属性为 cfg.train.cudnnbenchmark，从而使用训练配置中指定的设置。

##############################3. 设定随机种子 保证可复现性###############
    set_seed_and_print(cfg.seed) # 根据配置文件中的 seed 设置随机种子。这通常是为了保证实验的可复现性。

#############################4. 训练可视化####################
    if accelerator.is_main_process: # 检查当前进程是否为主进程（在分布式训练中，只有主进程会执行某些操作，例如数据可视化）。
        viz = vis_utils.get_visdom_connection(
            server=f"http://{cfg.viz_ip}", port=int(os.environ.get("VISDOM_PORT", 10088))
        ) # 如果是主进程，则连接到 Visdom 服务器，通常用于可视化训练过程中的损失、准确率等指标。

###########################5. 构建数据集#############################
    # Building datasets 调用 build_dataset 函数，构建训练和评估所需的数据集对象和数据加载器（dataloader 是训练集的数据加载器，eval_dataloader 是评估集的数据加载器
    dataset, eval_dataset, dataloader, eval_dataloader = build_dataset(cfg) # mix 混合的方式

    # to make accelerator happy 设置训练集的数据加载器，确保每个批次的大小都是完整的。drop_last=True 会丢弃最后一个不完整的批次。
    dataloader.batch_sampler.drop_last = True
    eval_dataloader.batch_sampler.drop_last = True

#################################6. 实例化模型############################
    # Instantiate the model 根据配置文件中的 MODEL 字段，使用 Hydra 的 instantiate 函数动态创建模型实例。_recursive_ 参数控制是否递归实例化模型中的子组件。
    model = instantiate(cfg.MODEL, _recursive_=False, cfg=cfg)
    model = model.to(accelerator.device)

    num_epochs = cfg.train.epochs # 500 从配置文件中获取训练的总 epoch 数。

    # Building optimizer ###############7. 构建优化器########################
    optimizer, lr_scheduler = build_optimizer(cfg, model, dataloader) # 使用配置文件中的参数和训练集数据构建优化器（optimizer）和
    # 学习率调度器（lr_scheduler）。优化器控制模型的权重更新，学习率调度器控制学习率的变化。

    ########################################################################################################################
    if cfg.train.resume_ckpt: # 检查恢复标志：首先检查 cfg.train.resume_ckpt 是否为 True，表示是否需要从上一个训练的检查点恢复模型的状态。
        # 第一次恢复（load_model_weights）：恢复的是模型model(如其名）的权重，用于加载神经网络的参数。
        accelerator.print(f"Loading ckpt from {cfg.train.resume_ckpt}") # 如果为 True，打印加载的检查点路径。
        model = load_model_weights(model, cfg.train.resume_ckpt, accelerator.device, cfg.relax_load) # 而 cfg.relax_load 是一个标志，表示是否松弛加载要求，可能允许加载不完全匹配的模型权重。
        # 如果为 True，通过调用 load_model_weights 函数来加载模型权重，这里使用了 accelerator.device 来确保加载到正确的设备上（例如 GPU 或 CPU）。

    ################################################9. Accelerator 预处理####################################
    # accelerator preparation 加速器准备。 通过 accelerator.prepare() 函数将模型、数据加载器、优化器和学习率调度器传入加速器。这会确保它们能够在多设备或分布式训练环境中正确运行。
    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(model, dataloader, optimizer, lr_scheduler)

    accelerator.print("length of train dataloader is: ", len(dataloader)) # 打印训练数据加载器的长度。
    accelerator.print("length of eval dataloader is: ", len(eval_dataloader))

    accelerator.print(f"dataloader has {dataloader.num_workers} num_workers") # 打印数据加载器的工作线程数。

############################################10. 训练状态初始化###########################
    start_epoch = 0

    stats = VizStats(TO_PLOT_METRICS) # 创建一个 VizStats 实例，TO_PLOT_METRICS 是一个定义要绘制的指标的常量。包含很多要绘制的内容，统计量loss等

    # CHECKPOINT RESUMING 检查点恢复。
    if cfg.train.auto_resume: # 检查是否启用了自动恢复。如果为 True，则表示训练将自动从最后一个检查点恢复。 第二次恢复（accelerator.load_state）：恢复的是整个训练过程的状态，包括模型的权重、优化器的状态、学习率调度器等。
        # if cfg.debug:
        #     import pdb;pdb.set_trace()

        last_checkpoint = find_last_checkpoint(cfg.exp_dir) # 通过 find_last_checkpoint 函数查找上一次训练的检查点，
        # cfg.exp_dir 是保存实验的目录

        try: # 尝试以下代码块，如果出错则进入 except 部分。
            resume_epoch = int(os.path.basename(last_checkpoint)[5:]) # 从检查点路径的文件名中提取 epoch 信息
            # （假设文件名的前 5 个字符后跟着数字表示 epoch）。例如，如果文件名为 ckpt_5，则提取 5 作为恢复的 epoch。
        except:
            resume_epoch = -1

        #################################12. 启动训练#############################
        if last_checkpoint is not None and resume_epoch > 0: # 如果找到了有效的检查点，并且恢复的 epoch 大于 0，进行恢复操作。
            accelerator.print(f"Loading ckpt from {last_checkpoint}")

            accelerator.load_state(last_checkpoint) # 加载检查点的状态。

            try:
                loaded_tdict = pickle.load(open(os.path.join(last_checkpoint, "tdict.pkl"), "rb")) # 从检查点加载训练字典（包含 epoch 等信息）。
                # 加载检查点目录中的 tdict.pkl 文件，它包含了训练的相关信息，比如上次训练的 epoch。
                start_epoch = loaded_tdict["epoch"] - 1  # + 1 # 从加载的字典中提取 epoch，并将 start_epoch 设置为上次训练的 epoch（减 1 是为了从下一个 epoch 开始
            except:
                start_epoch = resume_epoch - 1  # + 1

            try: # 尝试加载存储训练统计信息的文件 train_stats.jgz。
                stats = stats.load(os.path.join(last_checkpoint, "train_stats.jgz"))
            except:
                stats.hard_reset(epoch=start_epoch) # 如果加载统计数据失败，重置统计数据。
                accelerator.print(f"No stats to load from {last_checkpoint}")
        else:
            accelerator.print(f"Starting from scratch") # 如果没有找到检查点，则从头开始训练。

########################################1. 训练主循环#####################################
    for epoch in range(start_epoch, num_epochs): # 这行代码启动了一个 epoch 循环，从 start_epoch 开始，一直到 num_epochs
        stats.new_epoch() # 为当前的 epoch 创建一个新的统计记录。VizStats 类负责统计和可视化训练过程中的各种指标（如损失、准确率等），每个 epoch 开始时更新一次
        set_seed_and_print(cfg.seed + epoch * 1000) # 它根据给定的 cfg.seed 和当前 epoch 来设置随机种子。epoch * 1000 使得每个 epoch 的种子值不同，
        # 以确保每次训练时的随机性（例如初始化和数据加载的随机性）是唯一的。 打印当前使用的随机种子，以便调试和复现。

           ################################每隔 ckpt_interval 个 epoch 进行一次检查点保存：#################
        if (epoch != 0) and epoch % cfg.train.ckpt_interval == 0: # 只有当 epoch 不是第一个 epoch 且当前 epoch 是 cfg.train.ckpt_interval 的倍数时，才会执行保存检查点操作。
            ckpt_path = os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}") #根据当前的 epoch 创建一个检查点路径 ckpt_path。路径会保存到 cfg.exp_dir 目录下，文件名格式为 ckpt_000001（如果是第 1 个 epoch）
            accelerator.print(f"----------Saving the ckpt at epoch {epoch} to {ckpt_path}----------") # 打印一条消息，说明当前正在保存的检查点是针对哪个 epoch

            if accelerator.is_main_process: # accelerator.is_main_process 确保在分布式训练时，仅主进程（通常是 GPU 0）进行保存操作。这可以避免多次保存相同的检查点。
                accelerator.save_state(output_dir=ckpt_path, safe_serialization=False) # 存当前的训练状态（包括模型的权重、优化器的状态、学习率调度器等）。保存的目录是 ckpt_path，即当前 epoch 对应的路径。
                pickle.dump({"epoch": epoch, "cfg": cfg}, open(os.path.join(ckpt_path, "tdict.pkl"), "wb"))
                # 将当前的 epoch 和配置 cfg 保存到 tdict.pkl 文件中，以便在恢复训练时能够读取到这些信息。pickle 用于序列化 Python 对象
                stats.save(os.path.join(ckpt_path, "train_stats.jgz")) # 将训练过程中的统计信息（例如损失、精度等）保存到 train_stats.jgz 文件中。

        ####################################3. 强制评估（可选）#################################
        if cfg.force_eval: # cfg.force_eval 是一个配置项，表示是否强制执行评估步骤。如果为 True，则强制进行评估，即使在训练过程中。
            train_or_eval_fn( # 是用于训练或评估的函数。
                model,
                eval_dataloader,
                cfg,
                optimizer,
                stats,
                accelerator,
                lr_scheduler,
                training=False,
                epoch=epoch,
                viz=viz,
            )  # viz=viz)
            raise NotImplementedError # 抛出 NotImplementedError，表示此处代码块未实现。

#####################################4、调试模式下强制执行评估#########################
        if cfg.debug: # 如果配置项 cfg.debug 为 True，则进入调试模式。调试时会打印一些信息或执行一些额外的调试代码（如打印数据项等
            print("debugging")
            # dataset.__getitem__((0))
            # eval_dataset.__getitem__((0))
            # for hahaha in range(10):
            #    dataset.__getitem__((hahaha))
            if cfg.debugeval: # 如果 cfg.debugeval 为 True，则强制执行评估（与 cfg.force_eval 类似），调用 train_or_eval_fn 来评估模型。
                train_or_eval_fn(
                    model,
                    eval_dataloader,
                    cfg,
                    optimizer,
                    stats,
                    accelerator,
                    lr_scheduler,
                    training=False,
                    epoch=epoch,
                    viz=viz,
                )  # viz=viz)

        ##########################5. 强制测试（可选）###########################
        if cfg.force_test: # 是否强制进行测试步骤。测试是在每个 epoch 中强制执行的。
            test_imc(model, cfg, accelerator, epoch=epoch, print_detail=False) # 调用 test_imc 函数对模型进行 IMC 测试
            test_co3d(model, cfg, accelerator, epoch=epoch, print_detail=False)

        # Testing ################6. 评估 / 测试调度############################
        if (epoch != 0) and (epoch % cfg.test.test_interval == 0): # 每隔 test_interval 个 epoch 执行一次测试
            accelerator.print(f"----------Start to test at epoch {epoch}----------")
            # test_co3d(model, cfg, accelerator, epoch=epoch, print_detail=False) # 调用 test_co3d 函数对模型进行 3D 测试。
        elif (epoch != 0) and (epoch % cfg.train.eval_interval == 0):
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            train_or_eval_fn(
                model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=False, epoch=epoch
            )  # viz=viz)
        else:
            accelerator.print(f"----------Skip the test/eval at epoch {epoch}----------")

        # Training############################ 7. 训练 ##################################
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        train_or_eval_fn(
            model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True, epoch=epoch
        )  # viz=viz)

        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        ######################################8. 训练信息记录################################
        if accelerator.is_main_process: # 只有主进程（通常是 GPU 0）会执行下面的操作，避免多个进程重复操作。
            lr = lr_scheduler.get_last_lr()[0] # 获取当前的学习率。lr_scheduler.get_last_lr() 返回一个包含多个学习率的列表（适用于多学习率策略），[0] 是提取第一个学习率值
            accelerator.print(f"----------LR is {lr}----------")
            accelerator.print(f"----------Saving stats to {cfg.exp_name}----------") # 打印当前保存训练统计信息的路径（使用 cfg.exp_name 作为目录名
            stats.update({"lr": lr}, stat_set="train") # 更新训练统计数据，添加当前学习率 lr
            stats.plot_stats(viz=viz, visdom_env=cfg.exp_name) # 绘制训练过程中的统计信息，并将其发送到可视化工具（如 Visdom）。cfg.exp_name 用作 Visdom 环境的名称。
            accelerator.print(f"----------Done----------")
            # viz.save([cfg.exp_name])

 ####################################9. 最终保存检查点##########################
    accelerator.save_state(output_dir=os.path.join(cfg.exp_dir, f"ckpt_{epoch:06}"), safe_serialization=False) # 在每个 epoch 结束时保存当前的训练状态（包括模型权重、优化器、学习率调度器等）。
    # 保存的路径是 cfg.exp_dir 目录下，以 ckpt_{epoch:06} 命名 {epoch:06} 表示：    # epoch：要格式化的变量。    # :06：格式说明：    # 0：使用 0 进行填充（补全）。    # 6：总宽度为 6 位。
    return True

if __name__ == '__main__':
    cfg = OmegaConf.load('train.yaml')
    train_fn(cfg)
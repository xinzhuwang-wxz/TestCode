# TestCode
just some codes for test


一：环境搭建及修改


    https://github.com/Tom0126/ParticlePhysicsMLTutorial
    按上述链接bulid->git pull->source setup.h
  微调

    更新conda，torch；下载torch_geometric,torch_cluster
    目前使用的版本：
    PyTorch version: 2.1.2+cpu
    Torch Geometric version: 2.4.0
    Torch Cluster version: 1.6.3+pt21cpu
    下载cluster时，我使用的是：
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
    
    
    因为torch涉及多个库，以上安装好后，先讲下面的.py文件替换后，通过跑程序查看报错文件（log.err）有针对性进行安装与更新
二：Train.py

三：gravnet.py

四：loader.py

五：注意注意注意

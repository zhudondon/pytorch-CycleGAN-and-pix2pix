root
├─docs 文档
├─imgs 图片
├─models 模型
├─options 配置项
├─scripts 脚本，拉取数据
│  ├─edges
│  └─eval_cityscapes
│      └─caffemodel
└─util 工具











D:.
├─.idea
│  └─inspectionProfiles
├─data
├─datasets
│  ├─bibtex
│  └─horse2zebra
│      ├─testA
│      ├─testB
│      ├─trainA
│      └─trainB
├─docs
├─imgs
├─options
├─scripts
│  ├─edges
│  └─eval_cityscapes
│      └─caffemodel
└─util
文件夹 PATH 列表
卷序列号为 00000004 CE1D:B3F5
D:\A
无效的路径 - \A
没有子文件夹

PS D:\Users\Administrator\python\pytorch-CycleGAN-and-pix2pix> tree /f
文件夹 PATH 列表
卷序列号为 CE1D-B3F5
D:.
│  CycleGAN.ipynb
│  environment.yml
│  learn.md
│  LICENSE
│  pix2pix.ipynb
│  README.md
│  requirements.txt  依赖
│  test.py     测试
│  train.py    训练
│
├─data  数据，包含 拉取的
│      aligned_dataset.py 对齐数据集合
│      base_dataset.py 数据转换的基类 例如剪切，旋转，缩放等
│      colorization_dataset.py RGB与Lab颜色空间互相转换
│      image_folder.py
│      single_dataset.py  数据转换 单个
│      template_dataset.py  模板数据转换
│      unaligned_dataset.py  非对齐数据集合处理
│      __init__.py
│
├─datasets 数据集，包含下载
│  │  combine_A_and_B.py
│  │  download_cyclegan_dataset.sh
│  │  download_pix2pix_dataset.sh
│  │  make_dataset_aligned.py 数据集对齐处理
│  │  prepare_cityscapes_dataset.py 预处理
│  │
│  ├─bibtex BibTeX 是一种文件格式，也是一个制作这种文件的工具。这种文件用于描述和处理引用列表，通常情况下与LaTeX文档结合使用。
│  │      cityscapes.tex
│  │      facades.tex
│  │      handbags.tex
│  │      shoes.tex
│  │      transattr.tex
│  │
│  └─horse2zebra
│      ├─testA
│      ├─testB
│      ├─trainA
│      └─trainB
│
├─docs 文档
│      datasets.md
│      docker.md
│      Dockerfile
│      overview.md
│      qa.md
│      README_es.md
│      tips.md
│
├─imgs
│      edges2cats.jpg
│      horse2zebra.gif
│
├─models 模型
│      base_model.py
│      colorization_model.py
│      cycle_gan_model.py
│      networks.py
│      pix2pix_model.py
│      template_model.py
│      test_model.py
│      __init__.py
│
├─options 配置项
│      base_options.py
│      test_options.py
│      train_options.py
│      __init__.py
│
├─scripts shell脚本
│  │  conda_deps.sh
│  │  download_cyclegan_model.sh
│  │  download_pix2pix_model.sh
│  │  install_deps.sh
│  │  test_before_push.py
│  │  test_colorization.sh
│  │  test_cyclegan.sh
│  │  test_pix2pix.sh
│  │  test_single.sh
│  │  train_colorization.sh
│  │  train_cyclegan.sh
        get_data.py
        html.py
        image_pool.py
        util.py
        visualizer.py
        __init__.py














<img src="tests\backgrounds\tree_road_2.webp" alt="webp"  width="350"/> <img src="tests\objects\sbercat_book.png" alt="png"  width="250"/> <img src="tests\objects\pngtree_red_1.png" alt="png"  width="100"/> <img src="tests\objects\pngtree_red_2.png" alt="png"  width="100"/>

<img src="results\combined_image.png" alt="png"  width="390"/> <img src="results\combined_masks\depth_overlay_combined.png" alt="png"  width="390"/> 


Depth_Maker/
│
├── README.md
├── requirements.txt
├── .gitignore
├── depth_maker/
│   ├── __init__.py
│   ├── depth_maker_v1/  # first version maker
│   │   ├── __init__.py
│   │   ├── constructor_layered_image.py
│   │   ├── stylized_layered_masks.py
│   │   └── utils.py
│   └── depth_maker_v2/  # second version maker
│       ├── __init__.py
│       └── ...            
├── models/
│   ├── Depth_Anything_V2/
│   │   ├── __init__.py
│   │   └── ...            # other modules of Depth-Anything-V2
│   ├── Depth_ZOE/
│   │   ├── __init__.py
│   │   └── ...            # other modules of Depth-ZOE
│   └── ...
├── checkpoints/
│   ├── __init__.py
│   ├── Depth_Anything_V2/
│   │   └──depth_anything_v2_vitl.pth  # Model checkpoint
│   ├── Depth_ZOE/
│   │   └── depth_zoe.pth  # Model checkpoint
│   └── ...
├── tests/
│   ├── __init__.py
│   └── test_utils.py
└── notebooks/
    └── usage_examples.ipynb  # Future: Jupyter notebooks for showcasing usage

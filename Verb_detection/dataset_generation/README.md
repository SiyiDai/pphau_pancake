# Dataset Generation
This directory contains the scripts for generate the action detection dataset.

### Data Labeling
To label the image data, we have folders structured as follow:
- level 1: Action names
- level 2: Sequences
- level 3: Images

In each level, we have a csv file generated for debug convenience. For example, from bottom to top, ```decorate_sy0_0.csv``` contains timestamps, action labels and hand landmarks for images in sequence ```decorate_sy0_0```, ```data.csv``` under ```decorate``` combines all data from all sequences in ```decorate``` action of record ```sy0```. The top level data.csv, contains all data in record ```sy0``` and pad the non-detection/free-motion timestamps with all zero hand landmarks.

```
.
├── close
│   ├── data.csv
│   └── sy0_0
│       ├── close_sy0_0.csv
│       └── images
├── data.csv
├── data_int.csv
├── decorate
│   ├── data.csv
│   ├── sy0_0
│   │   ├── decorate_sy0_0.csv
│   │   └── images
│   └── sy0_1
│       ├── decorate_sy0_1.csv
│       └── images
├── flip
│   ├── data.csv
│   └── sy0_0
│       ├── flip_sy0_0.csv
│       └── images
├── move_object
│   ├── data.csv
│   └── sy0_0
│       ├── images
│       └── move_object_sy0_0.csv
├── open
│   ├── data.csv
│   └── sy0_0
│       ├── images
│       └── open_sy0_0.csv
├── other
│   ├── data.csv
│   ├── sy0_0
│   │   ├── images
│   │   └── other_sy0_0.csv
│   ├── sy0_1
│   │   ├── images
│   │   └── other_sy0_1.csv
│   ├── sy0_2
│   │   ├── images
│   │   └── other_sy0_2.csv
│   └── sy0_3
│       ├── images
│       └── other_sy0_3.csv
├── pick_up
│   ├── data.csv
│   ├── sy0_0
│   │   ├── images
│   │   └── pick_up_sy0_0.csv
│   ├── sy0_1
│   │   ├── images
│   │   └── pick_up_sy0_1.csv
│   ├── sy0_2
│   │   ├── images
│   │   └── pick_up_sy0_2.csv
│   └── sy0_3
│       ├── images
│       └── pick_up_sy0_3.csv
├── pour
│   ├── data.csv
│   └── sy0_0
│       ├── images
│       └── pour_sy0_0.csv
├── put_down
│   ├── data.csv
│   ├── sy0_0
│   │   ├── images
│   │   └── put_down_sy0_0.csv
│   ├── sy0_1
│   │   ├── images
│   │   └── put_down_sy0_1.csv
│   └── sy0_2
│       ├── images
│       └── put_down_sy0_2.csv
├── screw
│   ├── data.csv
│   ├── sy0_0
│   │   ├── images
│   │   └── screw_sy0_0.csv
│   └── sy0_1
│       ├── images
│       └── screw_sy0_1.csv
├── shovel
│   ├── data.csv
│   ├── sy0_0
│   │   ├── images
│   │   └── shovel_sy0_0.csv
│   └── sy0_1
│       ├── images
│       └── shovel_sy0_1.csv
└── squeeze
    ├── data.csv
    └── sy0_0
        ├── images
        └── squeeze_sy0_0.csv

```


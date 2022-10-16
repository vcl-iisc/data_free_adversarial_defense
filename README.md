## Pytorch implementation for DAD: Data-free Adversarial Defense at Test Time

### Method Overview
![technique overview](assets/dad-tech-overview.png)

### Evaluating Combined Performance (Correction + Detection):
``` ./scripts/combined.sh```

### Dependencies
- tqdm
- torch
- numpy
- torchattacks

### Citation:
If you use this code, please cite our work as:
```bibtex
    @inproceedings{
        nayak2021_DAD,
        title={DAD: Data-free Adversarial Defense at Test Time},
        author={Nayak, G. K., Rawal, R., and Chakraborty, A.},
        booktitle={IEEE Winter Conference on Applications of 
        Computer Vision},
        year={2022}
    }
```

### Acknowledgements

This repo borrows code from [Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation](https://github.com/tim-learn/SHOT) and [High Frequency Component Helps Explain the Generalization of Convolutional Neural Networks](https://github.com/HaohanWang/HFC)
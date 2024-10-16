# Comparative Analysis of Relevance Feedback Techniques for Image Retrieval

This GitHub repository contains the code to reproduce the comparison of four relevance feedback techniques: Rocchio, PicHunter$^\star$, Polyadic Query, and Linear Support Vector Machines. These techniques represent diverse strategies in relevance feedback, encompassing query vector modification, relevance probability estimation, adaptive similarity metrics, and classifier learning.

Note: PicHunter$^\star$ is our implementation of PicHunter, which is an enhancement of the original technique. It functions as PiHunter when using only positive feedback but also supports negative feedback. For further details, please refer to the paper.

The experiments conducted in this repository are described in the paper *"Comparative Analysis of Relevance Feedback Techniques for Image Retrieval."* This work provides insights into the effectiveness of each technique in improving image retrieval performance.





## Running the Experiments

To run all the experiments described in the paper, you can use the script located at `/src/exps_run_selected_methods.py`. This script contains all the tested cases as outlined in the paper. However, please note that running all the experiments may take a considerable amount of time (potentially 4-5 days!).

For an initial test, we recommend that you first select only the specific method, the number of seeds (for display initialization), and the type of feedback you are interested in. 

#### Download Data
The dataset used in the experiments is available on Zenodo and must be downloaded before running the above script. Please follow the link below to access and download the dataset:

[Download the dataset from Zenodo](https://doi.org/10.5281/zenodo.13941108)

After downloading, copy the files into the /data directory before proceeding with the experiments (do not extract the files).


## Experimental Analysis and Results

The notebook located at `/src/analysis_and_plot_results.ipynb` contains the analysis of the experimental results. In this notebook, you will find detailed insights and visualizations derived from the experiments.

Additionally, the folder `out_results` contains the precomputed results obtained from running the script `/src/exps_run_selected_methods.py`. These results can be used for further analysis or comparison without the need to rerun the experiments.

Here’s a section to encourage users to cite your paper:

---

## Citation

If you use this code or dataset in your research, please cite the following paper:

L. Vadicamo, F. Scotti, A. Dearle, R. Connor, *Comparative Analysis of Relevance Feedback Techniques for Image Retrieval*, in Proceedings of MMM 2025 (in press).

You can use the following BibTeX entry:

```bibtex
@inproceedings{vadicamo2025comparative,
  author    = {Lucia Vadicamo and Francesca Scotti and Alan Dearle and Richard Connor},
  title     = {Comparative Analysis of Relevance Feedback Techniques for Image Retrieval},
  booktitle = {Proceedings of MMM 2025},
  year      = {2025},
  note      = {in press}
}
```




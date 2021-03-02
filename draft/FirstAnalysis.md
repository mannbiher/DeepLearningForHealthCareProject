# Analysis of two topics

## Deep learning in Drug Discovery

learn a generating model that can produce target molecules with better properties from an input molecule.


### Data

- ZINC on github

### Task

reproduce the experiment of any of these molecule generation methods.



### Existing methods.

1. generate SMILES (line notation of chemical compound). Can be invalid.

2. Eliminate cycle using graph of substructures. Two level generating function to create new formula.

https://github.com/wengong-jin/icml18-jtnn4https://github.com/bowenliu16/rlgraphgenerationand

3. Enhancement graph to graph translational model. Can generate molecules with prespecified property.

https://github.com/wengong-jin/iclr19-graph2graph


4. Reinforcement learning with graph convolutional policy network

https://github.com/google-research/google-research/tree/master/moldqn

### Questions

How? the github code is already provided.

## NLP for Healthcare

ETL and NLP on clinical notes

### Data

- MIMIC 3 (Need access)
- COMPACT for aggregate analysis for clinical trial data

### Related Work

1. CAML (CNN with multi-label document classification) => Topic generation

https://www.aclweb.org/anthology/N18-1100.pdf

2. ElilE (Eligibility Criteria Information Extraction)
    - Entity and attr recognition
    - Negation detection
    - Relation extraction
    - Concept normalization and structuring

https://github.com/Tian312/EliIE

3. CFS (Convergent Focus Shift) pattern for drug trial pre and post
marketing. Drug with safety warning has different CFS pattern to
those without warnings. (Pattern recognition/clustering)

https://github.com/18dubu/ChunhuaLab_2016BBW

4. Identify overuse/unjustified use of exclusion of common eligibility
features (CEF) in mental disorder trials.

    No source code

5. Visual aggregate analysis of clinical trial analysis. 4 modules: 
    - Feature frequency analysis
    - Query Builder
    - Distribution Analysis
    - Visualization

    Analyzes
    - frequently used qualitativeand quantitative features for recruiting subjects for aselected medical condition
    
    - distribution of studyenrollment on consecutive value points or value inter-vals of each quantitative feature, and
    
    - distribution of studies on the boundary values, permissible value ranges, and value range widths of each feature.

    Didn't find any existing github repositories
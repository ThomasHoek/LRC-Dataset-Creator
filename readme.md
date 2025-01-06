# CCG to LRC for LangPro pipeline

## Directory and File Info

<table>
<thead>
    <tr>
    <th>Folder or Filename</th>
    <th>Description</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td><code>datasets_ccg</code></td>
        <td>Directory used to store CCG and SEN files</td>
    </tr>
    <tr>
        <td><code>datasets_original</code></td>
        <td>Directory used to store tsv and csv files</td>
    </tr>
    <tr>
        <td><code>LangProEdit</code></td>
        <td>Directory used for simple drag and drop replace for LangPro edits</td>
    </tr>
    <tr>
        <td><code>lex_KB</code></td>
        <td>Directory used for storing the created prolog knowledge bases</td>
    </tr>
    <tr>
        <td><code>lex_pairs</code></td>
        <td>Directory used for storing the created lexical pairs from NLI datasets</td>
    </tr>
    <tr>
        <td><code>lex_preds</code></td>
        <td>Directory used for storing the model predictions on the lexical pairs</td>
    </tr>
    </tr>
    <tr>
        <td><code>models</code></td>
        <td>Hidden directory used to store ML omdels</td>
    </tr>
    <tr>
        <td><code>results</code></td>
        <td>Overal directory for LangPro results with subdirectories for settings</td>
    </tr>
    <tr>
        <td><code>scripts_ccg</code></td>
        <td>Directory used to store scripts to compile CCG to lexical pairs</td>
    </tr>
    <tr>
        <td><code>scripts_clues</code></td>
        <td>Directory used for scripts to compile lexical pairs into lex_preds and lex_KB using <a href="https://aclanthology.org/2023.acl-long.308">Clues</a></td>
    </tr>
    <tr>
        <td><code>scripts_NLI</code></td>
        <td>Directory used for scripts to compile lexical pairs into lex_preds and lex_KB using <a href="https://huggingface.co/sileod/deberta-v3-base-tasksource-nli">NLI</a></td>
    </tr>
    <tr>
        <td><code>scripts_other</code></td>
        <td>Overal directory for debugging and standalone scripts.</td>
    </tr>
</tbody>
</table>

## Sub folder information

### Lexical Knowledge base info

```bash
lex_KB
└── [dataset]
    └── [model]
        ├── final                           # Contains the final predictions in Prolog files
        └── predictions                     # Contains the final predictions in TSV files (only NLI)

lex_pairs
└── [dataset]
    └── meta                                # Contains meta data about duplicate pairs indexes

lex_preds
└── [dataset]
    ├── [model]
    │   └── pred                            # Contains the predictions of the lex pairs
    │   
    │
    └── NLI
        ├── pred                            
        │   ├── full                            # Contains the predictions of each template
        │   └── inter                           # Contains concated predictions of each pair
        └── templates                       # Contains files of words inserted into the NLI templates
```

### Results

```bash
results
└── [dataset]                      
    │
    ├── abduction                           # Results for training Langpro with abduction
    │
    ├── base                                # Standard LangPro Training
    │
    ├── LRC                                 # Results LangPro which includes induced lexical relation
    │   │
    │   ├── [model]                             # Results of induced relations without index sorting
    │   └── [model]-index                       # Results of induced relations with index sorting
    │
    └── LRC_proba                           # Results LangPro which includes induced lexical relation
        ├── [model]
        └── [model]-index
```

## Scripts Files

```bash
scripts_ccg
├── ccg_class.py                        # Main class file which contains the underlying functionalities written
├── ccg_debug.py                        # Debug file to split lex_pairs and lex_preds into seperate files
├── ccg_main.py                         # Main file which parses a CCG file into lexical pairs
├── ccg_parse.py                        # Helper file to parse CCG prolog files into ccg_class
└── __init__.py
```

```bash
scripts_clues
├── 0.5.train_clues.py                  # Prerequisite file to train a Clues Model
├── 1_csv2pred_raw.py                   # Transforms the lex_pairs into lex_preds using clues.
├── 2_to_prolog_proba.py                # Transforms the lex_preds into a prolog KB using probabilities threshhold
├── 2_to_prolog.py                      # Transforms the lex_preds into a prolog KB
└── config.yaml                         # Config used for 0.5.train_clues.py
```

```bash
scripts_NLI
├── 1_make_templates.py                 # Transforms lex_pairs into lex_preds/templates
├── 2_SICK_NLI_predict.py               # Transforms lex_preds/templates into lex_preds
├── 2.3_inter_from_full.py              # debug script to turn an error script from lex_preds/full into lex_preds/inter
├── 2.5_test_decision_tree.py           # File to test LOOCV over every template, model not saved
├── 2.5_train_decision_tree.py          # File to train a decision tree model
├── 2.7_visualise_decision_tree.py      # File to create pdfs of the decision tree
├── 3_predict_using_tree_proba.py       # Transforms the lex_preds into a prolog KB using probabilities threshhold
├── 3_predict_using_tree.py             # Transforms the lex_preds into a prolog KB
├── results_test.txt                    # Results LOOVC
└── templates.json                      # Templates used by 1_make_templates.py
```

```bash
scripts_other
├── comment_SNLI_problems.py            # comments problematic SNLI problems, hardcoded.
├── merge_LRC_dataset.ipynb             # jupyter notebook used to create datasets_orginal/merged_LRC_words
├── merge_LRC_dataset_syn_extra.ipynb   # jupyter notebook used to create datasets_orginal/merged_LRC_words with experimental synonym dataset (NOT USED)
├── nlidata2prolog.py                   # Creates prolog sen and spl files from dataset_original in dataset_ccg. (New datasets have to be coded in)
├── seperate_results.py                 # Debug file to split the .log results into seperate files depending on [label]-label
├── snli_broken                         # simple txt file containing broken sentences / sentence pairs.
└── sort_file.py                        # Helper scripts of seperate_results.py to sort the problems in order.
```

## TODO's

1. make the word "None" workable, causes errors in Pandas for NaN.
2. Rewrite scripts_ccg/ccg_parse.py using push-down -> update ccg_class regex
3. Add relevant flowcharts for methods from paper into readme.
4. Add requirements.txt
5. Make produce more robust and add produce readme file.
6. ccg_main config file

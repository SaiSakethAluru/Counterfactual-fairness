# Counterfactual-Fairness
Implementation of "Counterfactual Fairness in Text Classification through Robustness"

# Summary
This paper studies counterfactual fairness in text classification, which asks the question: “How would the prediction change if the sensitive attribute referenced in the example were different?”

# Requirements 
Python - 3.6

### Libraries used: 
```
Pytorch==1.4.0 
Pandas==1.0.3 
Numpy==1.18.2
```

# Results
### Counterfactual token fairness gaps measured w.r.t. 35 training terms

| Model                  | Eval NT            | Sythetic NT         | Synthetic Toxic      |
|:----------------------:|:------------------:|:-------------------:|:--------------------:|
| Baseline               |  0\.116 \(0\.140\) | 0\.105  \(0\.180\)  | 0\.065\(0\.061\)     |
| Blind                  | 0\.00  \(0\.00\)   | 0\.00  \(0\.00\)    | 0\.00  \(0\.00\)     |
| CF Aug                 | 0\.116  \(0\.127\) | 0\.100  \(0\.226\)  | 0\.059  \(0\.022\)   |
| Clp\_nontoxic,lambda=1 | 0\.023  \(0\.012\) | 0\.0065  \(0\.015\) | 0\.0144 \(0\.007\)   |
| Clp, lambda=0\.05      | 0\.043  \(0\.071\) | 0\.010  \(0\.082\)  | 0\.012  \(0\.024\)   |
| CLP, lambda=1          | 0\.027  \(0\.007\) | 0\.012  \(0\.015\)  | 0\.0114  \(0\.007\)  |
| Clp, lambda=5          | 0\.012  \(0\.002\) | 0\.003  \(0\.004\)  | 0\.0036   \(0\.004\) |

### CTF gaps on held out identity terms for non-toxic examples from the evaluation set.

| Model                | CTF Gap: Held out terms |
|:--------------------:|:-----------------------:|
| Baseline             | 0\.193   \(0\.091\)     |
| Blind                | 0\.178   \(0\.09\)      |
| CF Aug               | 0\.207  \(0\.087\)      |
| CLP\_nontox,lambda=1 | 0\.121  \(0\.095\)      |
| Clp, lambda=0\.05    |  0\.091  \(0\.078\)     |
| Clp, lambda=1        | 0\.040  \(0\.084\)      |
| Clp, lambda=5        | 0\.044  \(0\.076\)      |

###   TNR and TPR gaps for different models

| Model             | TNR Gap            | TPR Gap            |
|:-----------------:|:------------------:|:------------------:|
| Baseline          | 0\.150  \(0\.084\) | 0\.272 \(0\.082\)  |
| Blindness         | 0\.163  \(0\.039\) | 0\.293  \(0\.114\) |
| Augmentation      | 0\.151  \(0\.065\) | 0\.253  \(0\.083\) |
| CLP Nontoxic, l=1 | 0\.156             | 0\.261             |
| CLP, l=0\.05      | 0\.157  \(0\.058\) | 0\.246  \(0\.078\) |
| CLP, l=1          | 0\.175  \(0\.039\) | 0\.224  \(0\.104\) |
| CLP, l=5          | 0\.163  \(0\.041\) | 0\.272  \(0\.112\) |


*The values in the parenthesis are those that are reported by the authors, and values outside the parenthesis are those that we have computed.*

# INSTRUCTIONS
To run the **baseline, blindness and augmentation models**, set the dataset file paths accordingly in load_data function. Set the variables use_clp and use_clp_nontoxic both to False.<br/>   
To run the **Counterfactual Logit Pairing model**, set the dataset file paths used for baseline. Set the variable use_clp to True and use_clp_nontoxic to False and set the hyperparameter lambda accordingly.<br/>   
To run the **Counterfactual Logit Pairing Nontoxic model**, set the dataset file paths used for baseline. Set the variable use_clp to False and use_clp_nontoxic to True and set the hyperparameter lambda accordingly.  


# NOTE
1. We used a custom CNN text classification model since the exact CNN architecture used by the authors is not mentioned in the paper.
2. The evaluation dataset used in the paper is private, and hence we have used the test dataset from Kaggle challenge.
3. The exact split of the identity tokens is not specified, hence we performed a random split, keeping the three bigram identity tokens as held out terms as mentioned in the paper. 

# Future Work
Try more complex CNN architectures for comparison.

# Team Members
Students from IIT Kharagpur:
1. Sai Saketh Aluru - 16CS30030 
2. Potnuru Anusha - 16CS30027
3. PVSL Hari Chandana - 16CS30026 
4. K Sai Surya Teja - 16CS30015 
5. Kaustubh Maloo - 15MA20019

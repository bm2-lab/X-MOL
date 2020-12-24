# X-MOL
large-scale pre-training for molecular understanding and diverse molecular analysis

Work-Flow:<br>
　　　　　　　　　　　　　　　|Molecular property prediction <br>
data　　　　—|　　　　　　　　|Drug-drug inteartion prediction <br>
transformer　—|——— X-MOL ———|Chemical reaction prediction <br>
computing　—|　　　　　　　|Molecule generation <br>
　　　　　　　　　　　　　　　|Molecule optimization <br>


## fine-tuning to prediction tasks
sh train_ft.sh <br>
sh train_lrtemb.sh (knowlegde embedding) <br>

## fine-tuning to generation tasks
sh train_ft.sh <br>

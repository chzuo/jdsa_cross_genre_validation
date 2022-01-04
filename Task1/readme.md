# Dataset
For dataset related to task 1, see [Data](Data/).  

The dataset uses the column format: 
- The first column is the word itself.
- The second column coarse hyperlink tags. Tag *I* means the word is within the hyperlink text. 
- The third column uses BIO-annotated tags. Tag *B-CHE* means the word is the beginning of the check-worthy claim, and *I-CHE* means the word is within the check-worthy claim.

Empty line separates sentences.  

(All numbers are replaced with the token *<NUM>*)

See for instance this sentence:
```
People	O	B-CHE
with	O	I-CHE
a	O	I-CHE
micropenis	O	I-CHE
have	O	I-CHE
a	O	I-CHE
penis	O	I-CHE
that	O	I-CHE
is	O	I-CHE
at	O	I-CHE
least	I	I-CHE
<NUM>	I	I-CHE
standard	I	I-CHE
deviations	I	I-CHE
smaller	O	I-CHE
than	O	I-CHE
the	O	I-CHE
average	O	I-CHE
penis	O	I-CHE
.	O	O
```

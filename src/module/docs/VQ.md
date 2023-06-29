Vector Quantisation (VQ)
=

# \_\_init\_\_
 |parameter|description|type|shape|
 |:---|:---|:---|:---|
 |`n_e` |number of vector in the CodeBook. | `int` |
 |`e_dim` |dimension of every vector in the CodeBook.|`int`|
 |`beta` |commitment Loss coefficient.|`float`|
 |`legacy` |always be False.| `bool`|
 |`log_choice`| log variation of choice when True.|`bool`|

# forward
 |parameter|description|type|shape|
 |:---|:---|:---|:---|
 |`z`|batched encoded features|`float tensor`|`(bs, e_dim)`|

 |output|description|type|shape|
 |:---|:---|:---|:---|
 |`z_q`|batched output codes|`float tensor`|`(bs, e_dim)`|
 |`loss`|embedding loss PLUS commitment loss|`float`||
 |`min_encoding_indices`|batched min index|`float tensor`|`(bs,)`|
 |`v`|variation of index in the batch|`float` or `None` when `log_choice` is `False`||

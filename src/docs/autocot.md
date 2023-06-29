# AutoCoT Configs & Description

## BaseConfig
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0.0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0.0`|`float`|

## KeyNetConfig
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`n_layer`|number of transformer blocks|`int`|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`model_type`|CoTPC model type|We use`'s+a'`| |use `kwargs`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0.0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0.0`|`float`|
 |`embd_pdrop`|embedding MLP dropout, always `0`|`float`|
 |`max_timestep`|
 |`use_skip_connection`|

## ActNetConfig
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`n_layer`|number of transformer blocks|`int`|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`model_type`|CoTPC model type|We use`'s+a+cot'`| |use `kwargs`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0.0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0.0`|`float`|
 |`key_states`|indicate form of key states|We use `'a'` only| |use `kwargs`|
 
## RecNetConfig
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`n_layer`|number of transformer blocks|`int`|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`model_type`|CoTPC model type|We use`'s'`| |use `kwargs`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0.0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0.0`|`float`|

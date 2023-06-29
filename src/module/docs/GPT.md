GPT
=
# GPTConfig
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |other| | | |use `kwargs`|

# BlocksWithCoT
`NOTICE` Parameters in config

 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`embd_pdrop`|embedding MLP dropout, always `0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0`|`float`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0`|`float`|
 |`model_type`|CoTPC model type|`'s','s+a','s+cot','s+a+cot'`| |use `kwargs`|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`key_states`|indicate form of key states|`str`, example: `'acd'` means you use 4 prompts in front of trajectory, and only predicted `#0`,`#1`,`#3` key states. We only use this feature to build ActNet, in order to receive predicted VQ feature| |use `kwargs`|
 |`key_state_loss`|indicate which layers are used for key states prediction (we do not use these features)| | | |
 |`len_key_states`|num of key states|`int`|||
 |`mask`|use for causal attention mask, can adjust according to input|`float tensor`| `(block_size, block_size)` ||
 |`n_layer`|number of transformer blocks|`int`|

# KeyNet
## \_\_init\_\_
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`state_dim`|dimension of (original) state vector|`int`|
 |`action_dim`|dimension of (original) action vector|`int`|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`embd_pdrop`|embedding MLP dropout, always `0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0`|`float`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0`|`float`|
 |`model_type`|CoTPC model type|`'s','s+a','s+cot','s+a+cot'`| |use `kwargs`|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`key_states`|indicate form of key states|`str`, example: `'acd'` means you use 4 prompts in front of trajectory, and only predicted `#0`,`#1`,`#3` key states. We only use this feature to build ActNet, in order to receive predicted VQ feature| |use `kwargs`|
 |`key_state_loss`|indicate which layers are used for key states prediction (we do not use these features)|
 |`len_key_states`|num of key states|`int`|
 |`mask`|use for causal attention mask, can adjust according to input|`float tensor`| `(block_size, block_size)` |
 |`n_layer`|number of transformer blocks|`int`|

# ActNet
## \_\_init\_\_
 |parameter|description|type|shape|other|
 |:---|:---|:---|:---|:---|
 |`state_dim`|dimension of (original) state vector|`int`|
 |`action_dim`|dimension of (original) action vector|`int`|
 |`n_embd`|dimension of embedding feature|`int`|
 |`n_head`|number of attention heads|`int`|
 |`embd_pdrop`|embedding MLP dropout, always `0`|`float`|
 |`resid_pdrop`|output projection MLP dropout in transformer block, always `0`|`float`|
 |`attn_pdrop`|MLP dropout in transformer block, always `0`|`float`|
 |`model_type`|CoTPC model type|`'s','s+a','s+cot','s+a+cot'`| |use `kwargs`|
 |`block_size`|we need to use mask, so we constraint input lens. When you use `action`, `block_size` is twice of state trajectory lens|`int`|
 |`key_states`|indicate form of key states|`str`, example: `'acd'` means you use 4 prompts in front of trajectory, and only predicted `#0`,`#1`,`#3` key states. We only use this feature to build ActNet, in order to receive predicted VQ feature| |use `kwargs`|
 |`key_state_loss`|indicate which layers are used for key states prediction (we do not use these features)|
 |`len_key_states`|num of key states|`int`|
 |`mask`|use for causal attention mask, can adjust according to input|`float tensor`| `(block_size, block_size)` |
 |`n_layer`|number of transformer blocks|`int`|

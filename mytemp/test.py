import torch

B = 2
T = 10 + 3
len_key_states = 3
n_head = 8

print((T - len_key_states) // 2)
r = torch.randint(0, (T - len_key_states) // 2, [B])[:, None] * 2
print('r =', r)

mask = torch.arange(0, T).repeat(B, 1) > r + len_key_states
print('mask =', mask)

key_state_mask = torch.zeros([B, T, T], dtype=torch.bool)
print(key_state_mask)

key_state_mask[:, :len_key_states, :] = mask[:, None, :].repeat(1, len_key_states, 1)
print(key_state_mask)

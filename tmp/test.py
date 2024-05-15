#%%

out = ''

for i in range(1, 11000):
    split = list(str(i))

    for j in split:
        if j == '8':
            out += 'SNOWMAN'
        else:
            out += j
    out += '-'

    if len(out) > 80031:
        break

print(len(out))
print(out)

# %%
sum = 0

for i in range(1, 2001):
    s = 77777 % i
    q = 77777 // i
    sum += s + q

print(sum)
# %%
out = []
i = 1

while True:
    if i % 3 == 0:
        for j in range(i):
            out.append('Z')
            out.append('E')
            out.append('R')
            out.append('O')
    elif i % 3 == 1:
        for j in range(i):
            out.append('O')
            out.append('N')
            out.append('E')
    else:
        for j in range(i):
            out.append('T')
            out.append('W')
            out.append('O')
    i += 1
    if len(out) == 16833:
        break

print(i)

#%%
print(len(out))
#%%
outout = ''
for i in range(1, 16834):
    if i == 500:
        outout += out[i-1]
    if i == 1500:
        outout += out[i-1]
    if i == 2500:
        outout += out[i-1]
    if i == 3500:
        outout += out[i-1]
    if i == 4500:
        outout += out[i-1]
    if i == 5500:
        outout += out[i-1]
    if i == 6500:
        outout += out[i-1]
    if i == 7500:
        outout += out[i-1]
    if i == 8500:
        outout += out[i-1]
    if i == 9500:
        outout += out[i-1]
    if i == 10500:
        outout += out[i-1]
    if i == 11500:
        outout += out[i-1]
    if i == 12500:
        outout += out[i-1]
    if i == 13500:
        outout += out[i-1]
    if i == 14500:
        outout += out[i-1]
    if i == 15500:
        outout += out[i-1]
    if i == 16500:
        outout += out[i-1]


print(outout)
# %%
print(len(outout))

# %%

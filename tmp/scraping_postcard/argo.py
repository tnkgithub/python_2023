# %%
def isValid(s: str) -> bool:
    stack = []
    for c in s:
        if c == "(":
            stack.append(")")
        elif c == "[":
            stack.append("]")
        elif c == "{":
            stack.append("}")
        elif not stack or stack.pop() != c:
            return False
    return not stack


# %%
def isValid2(s: str) -> bool:
    if len(s) % 2 != 0:
        return False

    for i in range(len(s) // 2):
        tmp = s
        s = s.replace("()", "").replace("[]", "").replace("{}", "")
        if tmp != "" and tmp == s:
            return False
    return True


# %%
s = "([]){}"
print(isValid2(s))  # True
# %%
s = "({)}"
print(isValid2(s))  # False

# %%

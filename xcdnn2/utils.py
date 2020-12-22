from typing import List

def subs_present(cs: List[str], s: str, at_start: bool = False) -> bool:
    # find the characters/substrings in the string
    # return True if at least one of the substring present in the string s
    for c in cs:
        if not at_start:
            if c in s:
                return True
        else:
            if c == s[:len(c)]:
                return True
    return False

def print_active_tensors(printout: bool = True) -> int:
    # NOTE: This function does not work if imported, so you have to copy and paste
    # this code to your main file in order to make it work
    import gc
    npresents = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if printout:
                    print(type(obj), obj.size(), obj.dtype)
                npresents += 1
        except:
            pass
    return npresents

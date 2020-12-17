from typing import List

def subs_present(cs: List[str], s: str) -> bool:
    # find the characters/substrings in the string
    # return True if at least one of the substring present in the string s
    for c in cs:
        if c in s:
            return True
    return False

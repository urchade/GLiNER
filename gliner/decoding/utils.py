def is_nested(idx1, idx2):
    # Return True if idx2 is nested inside idx1 or vice versa
    return (idx1[0] <= idx2[0] and idx1[1] >= idx2[1]) or (idx2[0] <= idx1[0] and idx2[1] >= idx1[1])


def has_overlapping(idx1, idx2, multi_label=False):
    # Check for any overlap between two spans
    if idx1[:2] == idx2[:2]:
        return not multi_label

    return not (idx1[0] > idx2[1] or idx2[0] > idx1[1])


def has_overlapping_nested(idx1, idx2, multi_label=False):
    # Return True if idx1 and idx2 overlap, but neither is nested inside the other
    if idx1[:2] == idx2[:2]:
        return not multi_label

    return not ((idx1[0] > idx2[1] or idx2[0] > idx1[1]) or is_nested(idx1, idx2))

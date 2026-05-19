import numpy as np

# matrice H pour Hamming (7,4) étendue en 8 bits (bit de parité ajouté)
H = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 0]
])

# positions des bits d'information (selon convention exercice)
DATA_POS = [0, 1, 2, 3]


def correct_block(block):
    r = np.array(block)

    # syndrome
    s = H @ r % 2

    # conversion syndrome -> position erreur
    syndrome_val = s[0]*1 + s[1]*2 + s[2]*4

    # correction si erreur simple détectée
    if 1 <= syndrome_val <= 7:
        r[syndrome_val - 1] ^= 1

    return r.tolist()


def extract_data(block):
    return [block[i] for i in DATA_POS]


def hamming748_decode(bits):
    result = []

    # traiter par blocs de 8 bits
    for i in range(0, len(bits), 8):
        block = bits[i:i+8]

        # si bloc incomplet (sécurité)
        if len(block) < 8:
            continue

        corrected = correct_block(block)
        data = extract_data(corrected)
        result.extend(data)

    return result

def test_hammingDecode():
    # Decoding when no errors leads to sequence recovering
    assert hamming748_decode([1, 1, 0, 1, 0, 0, 1, 0]) == [1, 1, 0, 1] 
    assert hamming748_decode([1, 1, 0, 0, 1, 1, 0, 0]) == [1, 1, 0, 0]
    assert hamming748_decode([1, 1, 1, 1, 1, 1, 1, 1]) == [1, 1, 1, 1]
    assert hamming748_decode([0, 1, 1, 1, 1, 0, 0, 0]) == [0, 1, 1, 1]
    assert hamming748_decode([0, 1, 1, 0, 0, 1, 1, 0]) == [0, 1, 1, 0]
    assert hamming748_decode([0, 0, 1, 1, 0, 0, 1, 1]) == [0, 0, 1, 1]
    assert hamming748_decode([0, 0, 1, 0, 1, 1, 0, 1]) == [0, 0, 1, 0]
    assert hamming748_decode([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]) == [0, 0, 1, 1, 0, 0, 1, 0]
    # Ensure that one error is detected, and corrected
    assert hamming748_decode([1, 1, 0, 1, 0, 0, 0, 0]) == [1, 1, 0, 1]
    assert hamming748_decode([1, 1, 0, 0, 1, 1, 0, 0]) == [1, 1, 0, 0]
    assert hamming748_decode([1, 0, 1, 1, 1, 1, 1, 1]) == [1, 1, 1, 1]
    assert hamming748_decode([0, 1, 1, 1, 1, 0, 0, 0]) == [0, 1, 1, 1]
    assert hamming748_decode([0, 1, 1, 0, 0, 1, 1, 0]) == [0, 1, 1, 0]
    assert hamming748_decode([0, 0, 1, 1, 0, 0, 1, 0]) == [0, 0, 1, 1]
    assert hamming748_decode([0, 0, 1, 0, 1, 1, 0, 1]) == [0, 0, 1, 0]    
    # Ensure that two errors cannot be corrected
    assert hamming748_decode([1, 0, 1, 1, 0, 0, 1, 0]) != [1, 1, 0, 1]
    assert hamming748_decode([1, 1, 1, 1, 1, 1, 0, 0]) != [1, 1, 0, 0]
    assert hamming748_decode([0, 1, 1, 0, 1, 1, 1, 1]) != [1, 1, 1, 1]
    assert hamming748_decode([1, 0, 1, 1, 1, 0, 0, 0]) != [0, 1, 1, 1]
    assert hamming748_decode([1, 1, 1, 1, 0, 1, 1, 0]) != [0, 1, 1, 0]
    assert hamming748_decode([0, 1, 0, 1, 0, 0, 1, 1]) != [0, 0, 1, 1]
    assert hamming748_decode([0, 1, 0, 0, 1, 1, 0, 1]) != [0, 0, 1, 0]




test_hammingDecode()
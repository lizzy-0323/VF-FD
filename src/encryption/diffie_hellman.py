"""
@Author: laziyu
@Date:2023-3-14
@Description: diffie hellman algorithm for psi
"""

import random
from hashlib import sha3_256

from data_loader.file_reader import read_dataset_from_csv
from utils.const import Q


def generate_keys(q=Q, seed=None):
    """Generate keys using a fixed seed for reproducibility"""
    if seed is not None:
        random.seed(seed)
    alpha = random.randint(1, q - 1)
    beta = random.randint(1, q - 1)
    return alpha, beta


def hash_number(x, q=Q):
    """Hash the data using sha3_256"""
    return int(sha3_256(str(x).encode()).hexdigest(), 16) % q


def hash_and_encrypt(data, key, q=Q):
    """Hash and encrypt the data"""
    hashed_and_encrypted = [(pow(hash_number(x, q), key, q)) for x in data]
    return hashed_and_encrypted


def decrypt(data, encrypt_intersection, double_encrypt_data):
    """Decrypt the data based on the intersection of encrypted values"""
    intersect_positions = []
    # Ensure double_encrypt_data and data have the same length
    if len(double_encrypt_data) != len(data):
        raise ValueError("Length mismatch between data and double_encrypt_data")
    for i, encrypted_value in enumerate(double_encrypt_data):
        if encrypted_value in encrypt_intersection:
            intersect_positions.append(i)
    if intersect_positions:
        return [data[position] for position in intersect_positions]
    return None


def double_encrypt(data, key, q=Q):
    """double encrypt"""
    return [(pow(x, key, q)) for x in data]


def get_encrypt_intersection(u_ab, u_ba):
    """Get the intersection of encrypted sets"""
    return set(u_ab) & set(u_ba)


def run_psi_protocol(A_data, B_data, q=Q, seed=None):
    # Step 1: Generate keys for A and B
    alpha, beta = generate_keys(q, seed)

    # Step 2: A hashes and encrypts its data
    U_A = hash_and_encrypt(A_data, alpha, q)
    # Step 3: B hashes and encrypts its data
    U_B = hash_and_encrypt(B_data, beta, q)
    # Step 4: B encrypts U_A and sends to A
    U_AB = double_encrypt(U_A, beta, q)

    # Step 5: A encrypts U_B and sends to B
    U_BA = double_encrypt(U_B, alpha, q)

    # Step 6: Calculate intersection
    intersection = get_encrypt_intersection(U_AB, U_BA)

    # Step 7: Decrypt
    result = decrypt(A_data, intersection, U_AB)

    return result


if __name__ == "__main__":
    # Example usage:
    q = 15485863  # Large prime number
    seed = 42  # Random seed for reproducibility

    # Example data for A and B
    data_1, data_2, data_3, data_4 = (
        read_dataset_from_csv("./data/client_1.csv"),
        read_dataset_from_csv("./data/client_2.csv"),
        read_dataset_from_csv("./data/client_3.csv"),
        read_dataset_from_csv("./data/client_4.csv"),
    )
    id_list1, id_list2, id_list3, id_list4 = (
        data_1.index.tolist(),
        data_2.index.tolist(),
        data_3.index.tolist(),
        data_4.index.tolist(),
    )
    # id_list1 = [1, 2, 3, 4, 5]
    # id_list2 = [2, 3, 5, 9, 10]
    # id_list1 = list(range(0, 500))
    # id_list2 = list(range(0, 500, 5))
    # print(len(id_list1))
    # print(len(id_list2))
    # ground_truth = set(id_list1) & set(id_list2)
    # print(len(ground_truth))
    # intersection = run_psi_protocol(id_list1, id_list2, q, seed)
    # print(len(intersection))
    intersection1 = run_psi_protocol(id_list2, id_list3, q, seed)
    print(len(intersection1))
    # print(intersection1)
    intersection2 = run_psi_protocol(id_list1, id_list4, q, seed)
    print(len(intersection2))
    # print(intersection2)
    intersection = run_psi_protocol(intersection1, intersection2, q, seed)
    ground_truth = set(id_list1) & set(id_list2) & set(id_list3) & set(id_list4)
    print("len Intersection of DH algorithm:", len(ground_truth))
    print("len Intersection of A and B's data:", len(intersection))


!pip install qiskit==0.46
!pip install qiskit-aer

from qiskit_aer import Aer
from qiskit import execute
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.compiler import transpile

import numpy as np
from PIL import Image

import cv2
import os
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import random
from math import log
from google.colab.patches import cv2_imshow
from tqdm import tqdm
import hashlib

def generate_random_bits(length):
    return np.random.randint(2, size=length)

# Function to prepare qubits based on random bit string
def prepare_qubits(bit_string):
    qubits = []
    for bit in bit_string:
        qc = QuantumCircuit(1, 1)
        if bit == 0:  # Prepare qubit in Z basis
            pass
        else:         # Prepare qubit in X basis
            qc.h(0)   # Apply Hadamard gate
        qubits.append(qc)
    return qubits

# Function to measure qubits in specified basis
def measure_qubits(qubits, basis):
    measured_bits = []
    for i, qc in enumerate(qubits):
        if basis[i] == 'Z':  # Measure qubit in Z basis
            qc.measure(0, 0)
        else:                # Measure qubit in X basis
            qc.h(0)
            qc.measure(0, 0)
        backend = Aer.get_backend('qasm_simulator')
        result = execute(qc, backend, shots=1).result()
        counts = result.get_counts()
        measured_bit = int(list(counts.keys())[0])  # Extract measured bit
        measured_bits.append(measured_bit)
    return measured_bits

# Function to perform basis reconciliation
def basis_reconciliation(basis_a, basis_b):
    # Compare bases and keep matching bits
    key_a = [bit_a for bit_a, bit_b in zip(basis_a, basis_b) if bit_a == bit_b]
    key_b = [bit_b for bit_a, bit_b in zip(basis_a, basis_b) if bit_a == bit_b]
    return key_a, key_b

# Function to perform bit sifting
def bit_sifting(key_a, key_b):
    # Sift bits based on indices agreed upon in basis reconciliation
    sifted_key_a = [bit_a for bit_a, bit_b in zip(key_a, key_b) if bit_a == bit_b]
    sifted_key_b = [bit_b for bit_a, bit_b in zip(key_a, key_b) if bit_a == bit_b]
    return sifted_key_a, sifted_key_b

# Function to perform privacy amplification
def privacy_amplification(key_a, key_b):
    final_key_a = key_a[:len(key_a)//2]
    final_key_b = key_b[:len(key_b)//2]
    return final_key_a, final_key_b

# Main function for BB84 QKD protocol
def bb84_qkd():
    # Step 1: Generate random bit string and bases for A
    random_bits_a = generate_random_bits(length=100)
    random_bases_a = generate_random_bits(length=100)

    # Step 2: Prepare qubits based on random bit string for A
    qubits_a = prepare_qubits(random_bits_a)

    # Step 3: Send qubits to Bob and randomly choose bases for measurement
    random_bases_b = generate_random_bits(length=100)

    # Step 4: B measures qubits in random bases
    measured_bits_b = measure_qubits(qubits_a, random_bases_b)

    # Step 5: A and B perform basis reconciliation
    key_a, key_b = basis_reconciliation(random_bases_a, random_bases_b)

    # Step 6: A and B perform bit sifting
    sifted_key_a, sifted_key_b = bit_sifting(key_a, key_b)

    # Step 7: A and B perform privacy amplification
    final_key_a, final_key_b = privacy_amplification(sifted_key_a, sifted_key_b)

    return final_key_a, final_key_b

if __name__ == "__main__":
    key_a, key_b = bb84_qkd()
    print("Final key A:", key_a)
    print("Final key B:", key_b)

def getImageMatrix(imageName):
    im = Image.open(imageName)
    pix = im.load()
    color = 1
    if type(pix[0,0]) == int:
      color = 0
    image_size = im.size
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1],color

def getImageMatrix_gray(imageName):
    im = Image.open(imageName).convert('LA')
    pix = im.load()
    image_size = im.size
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
                row.append((pix[width,height]))
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1],color

def ACMTransform(img, num):
    rows, cols, ch = img.shape
    n = rows
    img_ACM = np.zeros([rows, cols, ch])
    for x in range(0, rows):
        for y in range(0, cols):
            img_ACM[x][y] = img[(x+y)%n][(x+2*y)%n]
    return img_ACM

def ACMEncryption(imageName, key):
    img = cv2.imread(imageName)
    for i in range (0,key):
        img = ACMTransform(img, i)
    cv2.imwrite(imageName.split('.')[0] + "ACMencryption.png", img)
    return img

def ACMDecryption(imageName, key):
    img = cv2.imread(imageName)
    rows, cols, ch = img.shape
    dimension = rows
    decrypt_it = dimension
    if (dimension%2==0) and 5**int(round(log(dimension/2,5))) == int(dimension/2):
        decrypt_it = 3*dimension
    elif 5**int(round(log(dimension,5))) == int(dimension):
        decrypt_it = 2*dimension
    elif (dimension%6==0) and  5**int(round(log(dimension/6,5))) == int(dimension/6):
        decrypt_it = 2*dimension
    else:
        decrypt_it = int(12*dimension/7)
    for i in range(key,decrypt_it):
        img = ACMTransform(img, i)
    cv2.imwrite(imageName.split('_')[0] + "ACMDecryption.png",img)
    return img

secret_key = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]

# XOR each bit of the final key with the corresponding bit of the secret key
encrypted_key = [key_a[i] ^ secret_key[i % len(secret_key)] for i in range(len(key_a))]

# Convert the encrypted key to binary string
encrypted_key_bin = "".join(map(str, encrypted_key))

# Take the first 10 bits (3 digits in decimal) of the binary string
digit= encrypted_key_bin[:10]

# Convert the selected portion of the binary string to an integer (3-digit)
key_encryption = int(digit, 2)

image = "smveclogo"
ext = ".png"
key = key_encryption

img = cv2.imread(image + ext)
cv2_imshow(img)

ACMEncryptionIm = ACMEncryption(image + ext, key)
cv2_imshow(ACMEncryptionIm)

secret_key = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]

encrypted_key = [key_b[i] ^ secret_key[i % len(secret_key)] for i in range(len(key_b))]

encrypted_key_bin = "".join(map(str, encrypted_key))

digit= encrypted_key_bin[:10]

key_decryption = int(digit, 2)

ACMDecryptionIm = ACMDecryption(image + "ACMencryption.png", key_decryption)
cv2_imshow(ACMDecryptionIm)

def getImageMatrix_gray1(imageName):
    # Open the image
    img = Image.open(imageName)

    # Get the image size
    image_size = img.size

    # Create a grayscale image
    img_grayscale = img.convert('L')

    # Get the pixel values
    image_matrix = []
    for width in range(image_size[0]):
        row = []
        for height in range(image_size[1]):
            pix = img_grayscale.load()
            row.append(pix[width,height])
        image_matrix.append(row)

    # Return the image matrix and size
    return image_matrix, image_size

image = "smveclogo"
ext = ".png"
ImageMatrix,image_size = getImageMatrix_gray1(image+ext)
samples_x = []
samples_y = []
for i in range(1024):
  x = random.randint(0, int(image_size[0])-2)
  y = random.randint(0,image_size[1]-1)
  samples_x.append(ImageMatrix[x][y])
  samples_y.append(ImageMatrix[x+1][y])
plt.figure(figsize=(10,8))
plt.scatter(samples_x,samples_y,s=2)
plt.title('Adjacent Pixel Autocorrelation - Original Image', fontsize=20)
plt.show()

image = "smveclogoACMencryption"
ext = ".png"
ImageMatrix,image_size = getImageMatrix_gray1(image+ext)
samples_x = []
samples_y = []
print(image_size)
for i in range(1024):
  x = random.randint(0,image_size[0]-2)
  y = random.randint(0,image_size[1]-1)
  samples_x.append(ImageMatrix[x][y])
  samples_y.append(ImageMatrix[x+1][y])
plt.figure(figsize=(10,8))
plt.scatter(samples_x,samples_y,s=2)
plt.title('Adjacent Pixel Autocorrelation - ACM Encryption on Image', fontsize=20)
plt.show()

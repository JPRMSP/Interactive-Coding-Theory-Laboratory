import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# -------------------------------------------------------
# Utility: Finite Field GF(2) arithmetic for BCH/RS demo
# -------------------------------------------------------

def gf_add(a, b):
    return a ^ b

def gf_mul(a, b, prim=0x11d, field_charac=256):
    p = 0
    while b:
        if b & 1:
            p ^= a
        a <<= 1
        if a & field_charac:
            a ^= prim
        b >>= 1
    return p % field_charac

def rs_encode(msg, n=15, k=11, prim=0x11d):
    gen = [1]
    for i in range(n-k):
        gen = poly_mul(gen, [1, gf_pow(2, i, prim)], prim)
    padded = msg + [0]*(n-k)
    for i in range(k):
        coef = padded[i]
        if coef != 0:
            for j in range(len(gen)):
                padded[i+j] ^= gf_mul(coef, gen[j], prim)
    return msg + padded[k:]

def gf_pow(x, power, prim=0x11d):
    r = 1
    for _ in range(power):
        r = gf_mul(r, x, prim)
    return r

def poly_mul(p, q, prim=0x11d):
    result = [0]*(len(p)+len(q)-1)
    for i in range(len(p)):
        for j in range(len(q)):
            result[i+j] ^= gf_mul(p[i], q[j], prim)
    return result

# -------------------------------------------------------
# LDPC Construction: Simple Gallager (n, w_r, w_c)
# -------------------------------------------------------
def gallager_ldpc(n=24, w_r=3, w_c=6):
    H = np.zeros((w_c, n), dtype=int)
    rows = n // w_r
    for i in range(w_c):
        idx = np.random.choice(n, w_r, replace=False)
        for j in idx:
            H[i][j] = 1
    return H

def belief_propagation(H, y, iters=5):
    m, n = H.shape
    L = y.copy()
    Q = np.tile(L, (m, 1))
    R = np.zeros((m, n))
    for _ in range(iters):
        for i in range(m):
            for j in range(n):
                if H[i][j] == 1:
                    others = [Q[i][k] for k in range(n) if k != j and H[i][k] == 1]
                    prod = np.prod(np.tanh(np.array(others)/2))
                    R[i][j] = 2*np.arctanh(prod)
        for j in range(n):
            for i in range(m):
                if H[i][j] == 1:
                    Q[i][j] = L[j] + np.sum([R[k][j] for k in range(m) if k != i and H[k][j] == 1])
    return np.sign(Q.sum(axis=0))

# -------------------------------------------------------
# Space-Time Coding (Alamouti)
# -------------------------------------------------------
def alamouti_encode(x1, x2):
    return np.array([[x1, x2], [-np.conj(x2), np.conj(x1)]])

def alamouti_decode(H, Y):
    h1, h2 = H
    r1 = np.conj(h1)*Y[0] + h2*np.conj(Y[1])
    r2 = np.conj(h2)*Y[0] - h1*np.conj(Y[1])
    return r1, r2

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("🧪 Interactive Coding Theory Laboratory (ICT-Lab)")

tabs = st.tabs(["BCH / RS Codes", "LDPC Codes", "Tanner Graph", "EXIT Charts", "Space–Time Coding"])

# -------------------------------------------------------
# 1. BCH/RS TAB
# -------------------------------------------------------
with tabs[0]:
    st.header("Reed–Solomon Encoder (Demo)")
    msg = st.text_input("Enter message symbols (comma separated integers):", "1,2,3,4,5,6,7,8,9,10,11")
    msg = list(map(int, msg.split(",")))

    if st.button("Encode using RS (15,11)"):
        code = rs_encode(msg)
        st.write("Encoded Codeword:", code)

        fig, ax = plt.subplots()
        ax.stem(code)
        st.pyplot(fig)

# -------------------------------------------------------
# 2. LDPC TAB
# -------------------------------------------------------
with tabs[1]:
    st.header("LDPC Gallager Construction")

    n = st.slider("Block length n", 12, 60, 24)
    wr = st.slider("Row weight wr", 2, 6, 3)
    wc = st.slider("Number of rows wc", 4, 12, 6)

    if st.button("Generate LDPC Matrix"):
        H = gallager_ldpc(n, wr, wc)
        st.write("Parity-Check Matrix H:", H)

        y = np.random.randn(n)
        decoded = belief_propagation(H, y, iters=5)
        st.write("Decoded bits:", decoded)

# -------------------------------------------------------
# 3. TANNER GRAPH TAB
# -------------------------------------------------------
with tabs[2]:
    st.header("Tanner Graph Visualization")
    H = gallager_ldpc(12, 3, 4)
    G = nx.Graph()

    m, n = H.shape
    for i in range(m):
        for j in range(n):
            if H[i][j] == 1:
                G.add_edge(f"c{i}", f"v{j}")

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 4. EXIT CHARTS TAB (simple simulation)
# -------------------------------------------------------
with tabs[3]:
    st.header("EXIT Chart (Demo)")

    Ia = np.linspace(0, 1, 50)
    Ie = Ia**0.5
    fig, ax = plt.subplots()
    ax.plot(Ia, Ie)
    ax.set_xlabel("Ia")
    ax.set_ylabel("Ie")
    st.pyplot(fig)

# -------------------------------------------------------
# 5. STBC TAB
# -------------------------------------------------------
with tabs[4]:
    st.header("Alamouti STBC Simulation")

    x1, x2 = 1+1j, -1+0.5j
    S = alamouti_encode(x1, x2)

    H = np.array([1+0.2j, -0.4+1j])
    Y = H @ S + 0.1*(np.random.randn(2) + 1j*np.random.randn(2))

    r1, r2 = alamouti_decode(H, Y)

    st.write("Decoded Symbols (r1, r2):", r1, r2)

    fig, ax = plt.subplots()
    ax.scatter([r1.real, r2.real], [r1.imag, r2.imag])
    st.pyplot(fig)

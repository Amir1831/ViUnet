class Config():
    IMG_SIZE = (128,128)
    PATCH_SIZE = (16,16)
    # Number of patch in one sequence
    NUM_PATCH = int((IMG_SIZE[0] / PATCH_SIZE[0])**2)
    # Number of patch out after feature extracting
    T = int(NUM_PATCH / 4)
    # Number of attention head
    NUM_H = 4
    # Number of Transformer Block
    NUM_TR = 4
    #Out channels
    out0 = 3
    out1 = 8
    out2 = 16
    # Block Depth
    BlockDep = 2
    Loss_co = 100
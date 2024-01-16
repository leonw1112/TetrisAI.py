# Tetris played by AI
from nes import NES
import pyautogui as pag;
import sys; sys.path.append("..")
import tensorflow as tf

nes = NES("tetris.nes")
nes.run()

# Controls are:
# Up, Left, Down, Right: W, A, S, D
# Select, Start:  G, H
# A, B: P, L





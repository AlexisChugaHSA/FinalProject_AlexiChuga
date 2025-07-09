#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import os
import sys
# add the project root folder (two levels up) to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import streamlit as st
import pickle
from game_of_life import GameOfLife, plot_game_of_life


# configuration of streamlit
st.set_page_config(page_title="Conway's Game of Life", layout="centered")

# load the initial state
if 'game' not in st.session_state:
    pickle_file = os.path.join("..", "..", "data", "initial.pkl")
    with open(pickle_file, "rb") as f:
        state = pickle.load(f).tolist()
    st.session_state.game = GameOfLife.from_list(state)
    st.session_state.running = False

game = st.session_state.game

# buttons of control
col1, col2, col3 = st.columns(3)

if col1.button("Start"):
    st.session_state.running = True

if col2.button("Pause"):
    st.session_state.running = False

if col3.button("Reset"):
    # Vuelve a cargar el estado inicial
    pickle_file = os.path.join("..", "..", "data", "initial.pkl")
    with open(pickle_file, "rb") as f:
        state = pickle.load(f).tolist()
    st.session_state.game = GameOfLife.from_list(state)
    st.session_state.running = False

#
alive = game.population()
total = game.state.size
dead = total - alive

st.write(f"**Generation:** {game.generation}")
st.write(f"**Alive cells:** {alive}")
st.write(f"**Dead cells:** {dead}")

#
fig = plot_game_of_life(game)
st.pyplot(fig)

#
if st.session_state.running:
    game.evolve_parallel(num_threads=8)
    st.rerun()

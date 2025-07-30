# Student Class Allocation

A Python-based, fully automated student‑to‑class allocation system that uses network flow (Ford–Fulkerson) to automatically assign students to classes based on their time‑slot preferences, class capacity constraints, and a minimum satisfaction threshold.

## Features

- **Preference‑based assignment**: Honors each student’s top 5 time‑slot preferences first, then falls back to other slots if needed.
- **Capacity enforcement**: Ensures each class’s enrollment stays within its specified minimum and maximum.
- **Satisfaction guarantee**: Guarantees at least _K_ students get one of their top 5 choices, or reports failure.
- **Pure Python, zero dependencies**: Only uses the standard library.

## Academic Context

This project was completed as Assignment 2 for FIT2004 Data Structures and Algorithms (Semester 1, 2025). The context and the requirements of the functions were provided by the assignment brief.

# Experiment notebooks

Os notebooks são relatórios executáveis curtos. Abra-os na ordem abaixo:

1. `00_dataset.ipynb` — definição e inspeção do Shape–Color.
2. `01_sanity_check.ipynb` — baseline denso com seed 42.
3. `02_controls.ipynb` — color-only, shape-only e cor não correlacionada.
4. `03_multiseed_dynamics.ipynb` — dinâmica temporal em cinco seeds.
5. `04_one_shot_pruning.ipynb` — máscaras one-shot a 50%.
6. `05_sparsity_sweep.ipynb` — sweep one-shot em 50%, 80% e 90%.
7. `06_rewinding.ipynb` — máscara fixa, estados de pesos diferentes.
8. `07_imp.ipynb` — iterative magnitude pruning.

`inspect_shape_color_dataset.ipynb` e `run_imp_colab.ipynb` são notebooks
legados/de execução operacional. Os arquivos numerados acima formam a narrativa
experimental organizada.

Por padrão, os notebooks apenas leem outputs existentes. Para repetir um
experimento, altere explicitamente `RUN_EXPERIMENT = True` no notebook
correspondente. Treinamentos longos não são iniciados ao usar “Run All”.

# iGlobOpt
## Global Optimization with interval arithmetic ###

**Решение задач глобальной оптимизации с помощью методов интервального анализа**

## 1. Описание задачи
Данный проект направлен на поиск эффективных способов решения задач глобальной оптимизации на основе методов интервального анализа. В ходе реализации проекта требуется проведение следующего комплекса работ:
+ Реализация параллельного алгоритма глобальной оптимизации при помощи методов интервального оценивания
+ Исследование влияния точности интервального оценивания на скорость сходимости к глобальному оптимуму
+ Исследование влияния вычисления рекорда минимума оптимизируемой функции на скорость сходимости к глобальному оптимуму
+ Реализация алгоритмов в виде нескольких процессов с использование библиотеки MPI
+ Реализация алгоритма в виде нескольких потоковс использованием библиотеки OpenMP
+ Исследование возможностей распараллеливания алгоритма на современных графических ускорителях

## 2. Список тестовых функций
1. Ackley 1 function: [-32.0, 32.0] "argMin": [0.0] "globMinY": 0.0 AnyDim: True	Dim: 2
2. Ackley 2 function: [-32.0, 32.0] "argMin": [0.0] "globMinY": -200.0 AnyDim: True	Dim: 2
3. Ackley 3 function: [-32.0, 32.0] [-32.0, 32.0] "argMin": [-32.0, -32.0] "globMinY": 82.4617 AnyDim: False	Dim: 2
4. Ackley 4 function: [-32.0, 32.0] [-32.0, 32.0] "argMin": [-23.5625, -29.0625] "globMinY": -4.9999521818884851 AnyDim: False	Dim: 2
5. Adjiman function: [-1.0, 2.0] [-1.0, 1.0] "argMin": [2.0, 0.10578] "globMinY": -2.02181 AnyDim: False	Dim: 2
6. Alpine 1 function: [-10.0, 10.0] "argMin": [0.0] "globMinY": 0.0 AnyDim: True	Dim: 2
7. Alpine 2 function: [0.0, 10.0] [0.0, 10.0] [0.0, 10.0] "argMin": [7.9170526982459462172, 7.9170526982459462172, 4.81584] "globMinY": -17.212404253615528 AnyDim: False	Dim: 3
8. Brad function: [-0.25, 0.25] [0.01, 2.5] [0.01, 2.5] "argMin": [-0.249878, 2.49992, 2.49985] "globMinY": 6.9360214289615749 AnyDim: False	Dim: 3
9. Bartels Conn Function: [-10.0, 10.0] [-10.0, 10.0]  "argMin": [0.0, 0.0] "globMinY": 1.0 AnyDim: False	Dim: 2
10. Beale function: [-4.5, 4.5] [-4.5, 4.5]  "argMin": [3.0, 0.5] "globMinY": 0.0 AnyDim: False	Dim: 2
11. Biggs EXP2 Function: [0.0, 20.0] [0.0, 20.0]  "argMin": [1.0, 10.0] "globMinY": 0.0 AnyDim: False	Dim: 2
12. Biggs EXP3 Function: [0.0, 20.0] [0.0, 20.0] [0.0, 20.0]  "argMin": [1.0, 10.0, 5.0] "globMinY": 0.0 AnyDim: False	Dim: 3
13. Biggs EXP4 Function: [0.0, 20.0] [0.0, 20.0] [0.0, 20.0] [0.0, 20.0]  "argMin": [1.0, 10.0, 1.0, 5.0] "globMinY": 0.0 AnyDim: False	Dim: 4
14. Biggs EXP5 Function: [0.0, 20.0] [0.0, 20.0] [0.0, 20.0] [0.0, 20.0] [0.0, 20.0]  "argMin": [1.0, 10.0, 1.0, 5.0, 4.0] "globMinY": 0.0 AnyDim: False	Dim: 5
15. Biggs EXP6 Function: [-20.0, 20.0] [-20.0, 20.0] [-20.0, 20.0] [-20.0, 20.0] [-20.0, 20.0] [-20.0, 20.0]  "argMin": [1.0, 10.0, 1.0, 5.0, 4.0, 3.0] "globMinY": 0.0 AnyDim: False	Dim: 6
16. Bird Function: [-6.283185307179586476925286766559, 6.283185307179586476925286766559] [-6.283185307179586476925286766559, 6.283185307179586476925286766559]  "argMin": [4.70104, 3.15294] [-1.58214, -3.13024]  "globMinY": -106.764537 AnyDim: False	Dim: 2
17. Bohachevsky 1 Function: [-100.0, 100.0] [-100.0, 100.0]  "argMin": [0.0, 0.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
18. Bohachevsky 2 Function: [-100.0, 100.0] [-100.0, 100.0]  "argMin": [0.0, 0.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
19. Booth Function: [-10.0, 10.0] [-10.0, 10.0]  "argMin": [1.0, 3.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
20. Bohachevsky 3 Function: [-100.0, 100.0] [-100.0, 100.0]  "argMin": [0.0, 0.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
21. Box-Betts Quadratic Sum Function: [0.9, 1.2] [9.0, 11.2] [0.9, 1.2]  "argMin": [1.0, 10.0, 1.0] "globMinY": 0.0 AnyDim: False	Dim: 3
22. Branin RCOS Function: [-5.0, 10.0] [0.0, 15.0]  "argMin": [-3.1415926535897932384626433832795, 12.275] [3.1415926535897932384626433832795, 2.275] [9.4247779607693797153879301498385, 2.425]  "globMinY": 0.3978873 AnyDim: False	Dim: 2
23. Branin RCOS 2 Function: [-5.0, 10.0] [0.0, 15.0]  "argMin": [-3.196988423389338,12.526257883092258]  "globMinY": -0.179891239069905 AnyDim: False	Dim: 2
24. Brent Function: [-10.0, 10.0] [-10.0, 10.0]  "argMin": [-10.0, -10.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
25. Brown Function: [-1.0, 4.0]  "argMin": [0.0, 0.0]  "globMinY": 0.0 AnyDim: True	Dim: 2
26. Bukin 2 Function: [-15.0, -5.0] [-3.0, 3.0]  "argMin": [-10.0, 0.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
27. Bukin 4 Function: [-15.0, -5.0] [-3.0, 3.0]  "argMin": [-10.0, 0.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
28. Bukin 6 Function: [-15.0, -5.0] [-3.0, 3.0]  "argMin": [-10.0, 1.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
29. Camel Function – Three Hump Function: [-5.0, 5.0] [-5.0, 5.0]  "argMin": [-10.0, 1.0]  "globMinY": 0.0 AnyDim: False	Dim: 2
30. Camel Function – Six Hump Function: [-5.0, 5.0] [-5.0, 5.0]  "argMin": [-0.0898, 0.7126] [-0.0898, 0.7126]  "globMinY": -1.0316 AnyDim: False	Dim: 2
31. Chichinadze Function: [-30.0, 30.0] [-30.0, 30.0]  "argMin": [6.189866586965680, 0.5]   "globMinY": -42.94438701899098 AnyDim: False	Dim: 2
32. Chung Reynolds function: [-100.0, 100.0]  "argMin": [0.0, 0.0]   "globMinY": 0.0 AnyDim: True	Dim: 2
33. Colville Function: [-10.0, 10.0] [-10.0, 10.0] [-10.0, 10.0] [-10.0, 10.0] "argMin": [1.0, 1.0, 1.0, 1.0]   "globMinY": 0.0 AnyDim: False	Dim: 4
34. Complex function: [-2.0, 2.0] [-2.0, 2.0] "argMin": [1.0, 0.0]   "globMinY": 0.0 AnyDim: False	Dim: 2
35. Cosine Mixture function: [-1.0, 1.0] [-1.0, 1.0] "argMin": [0.0, 0.0]   "globMinY": -0.2 AnyDim: False	Dim: 2
36. Cross In Tray function: [-15.0, 15.0] [-15.0, 15.0] "argMin": [1.349406608602084, 1.349406608602084] [-1.349406608602084, -1.349406608602084]   "globMinY": -2.062611870822739 AnyDim: False	Dim: 2
37. Cross-Leg function: [-10.0, 10.0] [-10.0, 10.0] "argMin": [0.0, 0.0]    "globMinY": -1.0 AnyDim: False	Dim: 2
38. Cube function: [-10.0, 10.0] [-10.0, 10.0] "argMin": [1.0, 1.0]    "globMinY": 0.0 AnyDim: False	Dim: 2
39. Deb 1 function: [-1.0, 1.0] "argMin": [-0.9] [-0.7] [0.7] [0.9]  "globMinY": -1.0 AnyDim: True	Dim: 2
40. Davis's function: [-100.0, 100.0] [-100.0, 100.0] "argMin": [0.0, 0.0]    "globMinY": 0.0 AnyDim: False	Dim: 2
41. Deckkers-Aarts function: [-20.0, 20.0] [-20.0, 20.0] "argMin": [-0.000532865524291992, 14.9482715129852]  "globMinY": -24776.472160967165 AnyDim: False	Dim: 2
42. Dixon & Price function: [-10.0, 10.0] [-10.0, 10.0] "argMin": [1.0, 0.70710678118654752440084436210485] "globMinY": 0.0 AnyDim: False	Dim: 2
43. Dolan function: [-100.0, 100.0] [-100.0, 100.0] [-100.0, 100.0] [-100.0, 100.0] [-100.0, 100.0] "argMin": [98.964258312237106, 100.0, 100.0, 99.224323672554704, -0.249987527588471] "globMinY": -529.8714387324576 AnyDim: False	Dim: 5
44. Drop-Wave function: [-5.12, 5.12] [-5.12, 5.12]  "argMin": [0.0, 0.0] "globMinY": -1.0 AnyDim: False	Dim: 2
45. Rosenbrock function 5: [-30.0, 30.0] ... [-30.0, 30.0]   "argMin": [1.0...] "globMinY": 0.0 AnyDim: True	Dim: 5
46. Rosenbrock function 10: [-30.0, 30.0] ... [-30.0, 30.0]   "argMin": [1.0...] "globMinY": 0.0 AnyDim: True	Dim: 10
47. Rosenbrock function 15: [-30.0, 30.0] ... [-30.0, 30.0]   "argMin": [1.0...] "globMinY": 0.0 AnyDim: True	Dim: 15
58. Rosenbrock function 20: [-30.0, 30.0] ... [-30.0, 30.0]   "argMin": [1.0...] "globMinY": 0.0 AnyDim: True	Dim: 20
59. Rosenbrock Modified function: [-2.0, 2.0] [-2.0, 2.0]   "argMin": [-0.9, -0.95] "34.371238966161798": 0.0 AnyDim: False	Dim: 2



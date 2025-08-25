"""Модуль, содержащий модель для асинхронного процесса."""

import numpy as np

class AsyncProcessModel():
    """Класс для вычиления характеристик асинхронного процесса."""
    def __init__(
            self, 
            K: int, 
            M: int, 
            m: list[list[float]], 
            sigma: list[float], 
            a: list[float], 
            v: float
    ) -> None:
        """Инициализация асинхронной модели.

        Parameters:
        ----------
            K: количество типов частиц
            M: количество ячеек
            m: мю
            sigma: сигма
            a: вероятность принадлежности ячейки к k-ому типу
            v: средняя скорость частиц
        """

        self.K = K
        self.M = M
        self.m = np.array([np.array(arr) for arr in m])
        self.sigma = np.array(sigma)
        self.a = np.array(a)
        self.v = v

        self._validate_values()
    
    def compute(self, hmax: int) -> tuple[np.ndarray, float]:
        """Функция для расчета среднего числа сводобных ячеек и плотности потока.

        Parameters:
        ----------
            hmax: максимальное значение h

        Returns:
        -------
            : среднее число свободных ячеек и плотность
        """
        R = self._compute_r()
        sigma_mults = self._compute_sigma_mult()
        C = self._compute_const(
            R=R,
            delimiters=sigma_mults
        )
        P = self._compute_probability(
            constants=C,
            delimiters=sigma_mults,
            R=R,
            hmax=hmax
        )
        l = np.zeros(self.K)

        for k in range(self.K):
            summa = 0
            for h in range(1, self.M):
                summa += (P[k, h] * h)
            
            l[k] = summa + (P[k, self.M] * self.M) / (1 - R[k]) * (P[k, self.M] * R[k]) / ((1 - R[k]) ** 2)
        
        rho = 1 / (1 + sum([self.a[k] * l[k] for k in range(self.K)]))

        return l, rho
    
    def _compute_probability(
            self, 
            constants: np.ndarray, 
            delimiters: np.ndarray, 
            R: np.ndarray, 
            hmax: int = 30
    ):
        """Вычисляет стационарные вероятности.

        Parameters:
        ----------
            constants: значения констант для расчета
            delimiters: значения произведений µ для знаменателя
            R: значения R для расчета
            hmax: максимальное значение h

        Returns:
        -------
            : стационарные вероятности P
        """
        P = np.zeros((self.K, hmax + 1))
        

        for k in range(self.K):
            C = constants[k]
            P[k, 0] = C

            for h in range(1, self.M + 1):
                P[k, h] = ((self.v ** h) * C) / delimiters[k, h - 1]
            
            for h in range(self.M + 1, hmax + 1):
                P[k, h] = P[k, self.M] * R[k] ** (h - self.M)
        
        return P

    def _compute_const(self, R: np.ndarray, delimiters: np.ndarray) -> np.ndarray:
        """Вычисляет значения констант C для расчета.

        Parameters:
        ----------
            R: значения R для расчета
            delimiters: значения произведений µ для знаменателя

        Returns:
        -------
            : значения констант для расчета
        """

        C = np.zeros(self.K)

        for k in range(self.K):
            curr_part = 1
            for h in range(1, self.M):
                curr_part = curr_part + (self.v ** h) / delimiters[k, h - 1]
            curr_part = curr_part + ((self.v ** h) / delimiters[k, h - 1]) / (1 - R[k])
            C[k] = 1 / curr_part
        
        return C
    
    def _compute_r(self) -> np.ndarray:
        """Вычисляет значения R для каждого k.

        Здесь k получается из массива по k - 1 (для k = 1 в формуле получать R[0]).

        Returns:
        -------
            : массив R, содержащий Rk для последующего расчета
        """
        R = self.v / self.sigma
        return R

        
    def _compute_sigma_mult(self) -> np.ndarray:
        """Считает произведения µ для дальнейшего расчета P.

        Returns:
        -------
            : матрица со значениями произведений µ
        """
        delimiters = np.zeros((self.K, self.M))

        for k in range(0, self.K):
            delimiters[k, 0] = self.m[k, 0]

            for h in range(1, self.M):
                delimiters[k, h] = delimiters[k, h - 1] * self.m[k, h]
        
        return delimiters
    
    def _validate_values(self) -> None: 
        """Валидация полей, поступивших на вход."""
        assert self.v < min(self.sigma)
        assert sum(self.a) == 1

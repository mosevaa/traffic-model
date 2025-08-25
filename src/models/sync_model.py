"""Модуль, содержащий модель для расчета синхронного процесса."""

import numpy as np

class SyncProcessModel:
    """Класс для вычисления характеристик синхронной модели."""
    def __init__(
            self,
            K: int,
            M: int,
            p: list[list[float]],
            sigma: list[float],
            a: list[float],
            v: float
    ) -> None:
        """Инициализация параметров для расчета

        Parameters:
        ----------
            K: количество типов ячеек
            M: количество ячеек
            p: вероятности перемещения ячеек 
               (Здесь k и h получается из массива по 
               k - 1 и h - 1 (для k = 1, h= 1 в формуле получать 
               p[0, 0]).)
            sigma: значения σ (Здесь k получается из массива по 
               k - 1 (для k = 1 в формуле получать R[0]).)
            a: вероятность принадлежности ячейки к k-ому типу
               (индексация k -> (k - 1))
            v: средняя скорость частиц
        """
        self.K = K
        self.M = M
        self.p = np.array([np.array(arr) for arr in p])
        self.sigma = np.array(sigma)
        self.a = np.array(a)
        self.v = v

        self._validate_values()

    def compute(self, hmax: int) -> tuple[np.ndarray[float], float]:
        """Функция для расчета среднего числа свободных ячеек и плотности потока.

        Parameters:
        ----------
            hmax: максимальное значение h

        Returns:
        -------
            : среднее число свободных ячеек и плотность
        """

        P = self._compute_probabilities(hmax=hmax)
        R = self._compute_r()

        l = np.zeros(self.K)

        for k in range(self.K):
            summa = 0
            for h in range(1, self.M):
                summa += (P[k, h] * h)
            
            l[k] = summa + ((P[k, self.M] * self.M) / (1 - R[k])) + ((P[k, self.M] * R[k]) / ((1 - R[k]) ** 2))
        
        rho = 1 / (1 + sum([self.a[k] * l[k] for k in range(self.K)]))

        return l, rho

    def _compute_probabilities(
            self,
            hmax: int,

    ):
        """Вычисление значений стационарных вероятностей.

        Индексация: k -> (k - 1), h -> h

        Parameters:
        ----------
            hmax: максимальное значение h

        Returns:
        -------
            : значения стационарных вероятностей
        """
        P = np.zeros((self.K, hmax + 1))
        R = self._compute_r()
        q_mults = self._compute_q_mult()
        p_mults = self._compute_p_mult()
        constants = self._compute_constants(
            R=R,
            mult_p=p_mults,
            mult_q=q_mults
        )
        w = 1 - self.v

        for k in range(self.K):
            P[k, 0] = constants[k]
            P[k, 1] = (constants[k] * self.v) / (w * self.p[k, 0])

            for h in range(2, self.M + 1):
                P[k, h] = (constants[k] * (self.v ** h) * q_mults[k, h - 1]) / ((w ** h) * p_mults[k, h - 1])
            
            for h in range(self.M + 1, hmax + 1):
                P[k, h] = P[k, self.M] * (R[k] ** h - self.M)
        
        return P

    def _compute_constants(
            self,
            R: np.ndarray,
            mult_p: np.ndarray,
            mult_q: np.ndarray
    ) -> np.ndarray:
        """Вычисление значений константы C для дальнейших расчетов.

        Индексация: k -> (k - 1), h -> h

        Parameters:
        ----------
            R: значения массива R
            mult_p: произведения p для расчета
            mult_q: произведения q для расчета

        Returns:
        -------
            : значения констант для каждого k
        """
        constants = np.zeros(self.K)
        w = 1 - self.v

        for k in range(self.K):
            curr_part = 1 + (self.v / (w * self.p[k, 0]))
            
            for h in range(2, self.M):
                curr_part += ((self.v ** h) * mult_q[k, h - 1]) / ((w ** h) * mult_p[k, h - 1])
                
            curr_part = curr_part + ((self.v ** self.M) * mult_q[k, self.M - 1]) / ((w ** (self.M)) * mult_p[k, self.M - 2]) / (1 - R[k])
            
            constants[k] = 1 / curr_part

        return constants
        
    def _compute_q_mult(self) -> np.ndarray:
        """Вычисление произведений q для дальнейшего счета.

        Индексация: k -> (k - 1), h -> (h - 1)

        Returns:
        -------
            : значения произведений q для дальнейшего расчета
        """
        q = self._compute_q()
        mult_q = np.zeros((self.K, self.M))

        for k in range(self.K):
            mult_q[k, 0] = q[k, 0]
            for h in range(1, self.M - 1):
                mult_q[k, h] = mult_q[k, h - 1] * q[k, h]
        
        return mult_q

    def _compute_p_mult(self):
        """Вычисление произведений вероятностей для дальнейшего счета.

        Индексация: k -> (k - 1), h -> (h - 1)

        Returns:
        -------
            : значения произведений вероятностей для дальнейшего расчета
        """
        mult_p = np.zeros((self.K, self.M))

        for k in range(self.K):
            mult_p[k, 0] = self.p[k, 0]

            for h in range(1, self.M):
                mult_p[k, h] = mult_p[k, h -1] * self.p[k, h]
        
        return mult_p

    def _compute_q(self) -> np.ndarray:
        """Вычисляет значения q для каждого k.

        Индексация: k -> (k - 1), h -> (h - 1)

        Returns:
        -------
            : значения массива q для дальнейшего расчета.
        """
        q = np.zeros((self.K, self.M))

        for k in range(self.K):
            for h in range(self.M):
                q[k, h] = 1 - self.p[k, h]
        
        return q

    def _compute_r(self):
        """Вычисляет значения R для каждого k.

        Здесь k получается из массива по k - 1 (для k = 1 в формуле получать R[0]).

        Returns:
        -------
            : массив R, содержащий Rk для последующего расчета
        """
        R = np.zeros(self.K)
        w = 1 - self.v
        for k in range(self.K):
            R[k] = (self.v * (1 - self.sigma[k])) / (w * self.sigma[k])
        
        return R

    #TODO: вынести валидацию в общий класс
    def _validate_values(self) -> None:
        assert sum(self.a) == 1
        assert self.v < min(self.sigma)



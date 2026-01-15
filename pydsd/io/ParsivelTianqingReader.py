# -*- coding: utf-8 -*-
import io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ..DropSizeDistribution import DropSizeDistribution
from . import common


def read_parsivel_tianqing(filename):
    """
    Takes a filename pointing to a parsivel tianqing csv file and returns
    a drop size distribution object.

    Usage:
    dsd = read_parsivel_tianqing(filename)

    Returns:
    DropSizeDistrometer object

    """
    reader = ParsivelTianqingReader(filename)
    dsd = DropSizeDistribution(reader)
    return dsd


class ParsivelTianqingReader(object):

    """
    Reader for Parsivel Tianqing format.

    Each row represents ONE particle.
    The PSD code m = j * 32 + i encodes velocity bin j and diameter bin i.
    """

    def __init__(self, filename, dt=60.0):
        self.filename = filename
        self.rain_rate = []
        self.Z = []
        self.num_particles = []
        self._base_time = []

        self.nd = []
        self.vd = []
        # self.raw = []
        self.code = []
        self.time = []

        self.ndt = []

        # self.pcm = np.reshape(self.pcm_matrix, (32, 32))

        self._read_file()
        self._prep_data()

        self.bin_edges = np.hstack(
            (0, self.diameter["data"] + np.array(self.spread["data"]) / 2)
        )

        self.bin_edges = common.var_to_dict(
            "bin_edges", self.bin_edges, "mm", "Bin Edges"
        )

        # self._apply_pcm_matrix()

    def _read_file(self):
        df = pd.read_csv(
            self.filename,
            usecols=["Datetime", "V13205", "V13206", "Q13206"],
            parse_dates=["Datetime"],
        )

        # 仅保留通过质控的数据
        df = df[df["Q13206"] == 0]

        Ts = 60.0  # s
        S = 54.0e-4  # m^2
        vel = self.velocity["data"]  # (32,)

        grouped = df.groupby("Datetime")

        for t, g in grouped:
            # 每个时次初始化 32 个粒径通道的数浓度
            nDi = np.zeros(32, dtype=float)

            for Aij in g["V13205"].values:
                j = Aij // 32  # 速度通道
                i = Aij % 32  # 粒径通道

                # 合法性检查（非常重要）
                if 0 <= j < 32 and 0 <= i < 32:
                    nDi[i] += Aij / (vel[j] * Ts * S)

            self.nd.append(nDi)
            vd = np.zeros(32, dtype=float)
            self.vd.append(vd)

            g_num = g["V13206"].sum()
            self.num_particles.append(g_num)
            self.time.append(t)


    def _raw_to_nd(self):
        Nd_all = []

        D = np.array(self.diameter["data"])
        dD = np.array(self.spread["data"])
        V = np.array(self.velocity["data"])
        A = np.array(self.sampling_area["data"])

        for raw in self.filtered_raw:
            Nd = np.zeros(32)

            for i in range(32):
                if dD[i] <= 0:
                    continue

                for j in range(32):
                    if V[j] > 0:
                        Nd[i] += raw[j, i] / (A[i] * V[j] * self.dt)

                Nd[i] /= dD[i]

            Nd_all.append(Nd)

        self.nd = np.array(Nd_all)

    def _prep_data(self):
        self.fields = {}

        self.fields["rain_rate"] = common.var_to_dict(
            "Rain rate", np.ma.array(self.rain_rate), "mm/h", "Rain rate"
        )
        self.fields["reflectivity"] = common.var_to_dict(
            "Reflectivity",
            np.ma.masked_equal(self.Z, -9.999),
            "dBZ",
            "Equivalent reflectivity factor",
        )
        self.fields["Nd"] = common.var_to_dict(
            "Nd",
            np.ma.masked_equal(self.nd, np.power(10, -9.999)),
            "m^-3 mm^-1",
            "Liquid water particle concentration",
        )
        self.fields["Nd"]["data"].set_fill_value(0)

        self.fields["num_particles"] = common.var_to_dict(
            "Number of Particles",
            np.ma.array(self.num_particles),
            "",
            "Number of particles",
        )
        self.fields["terminal_velocity"] = common.var_to_dict(
            "Terminal Fall Velocity",
            np.array(
                self.vd[0]
            ),  # Should we do something different here? Don't think we want the time series.
            "m/s",
            "Terminal fall velocity for each bin",
        )

        try:
            self.time = self._get_epoch_time()
        except:
            self.time = {
                "data": np.array(self.time, dtype=float),
                "units": None,
                "title": "Time",
                "full_name": "Native file time",
            }
            print("Conversion to Epoch Time did not work.")

    def _get_epoch_time(self):
        """
        Convert self.time (list of datetime.datetime) to Epoch time
        using package standard.
        """
        epoch = datetime.utcfromtimestamp(0)

        time_secs = np.ma.array(
            [(t - epoch).total_seconds() for t in self.time]
        )

        time_secs.set_fill_value(1e20)

        return {
            "data": time_secs,
            "units": common.EPOCH_UNITS,
            "standard_name": "Time",
            "long_name": "Time (UTC)",
        }

    diameter = common.var_to_dict(
        "diameter",
        np.array(
            [
                0.06,
                0.19,
                0.32,
                0.45,
                0.58,
                0.71,
                0.84,
                0.96,
                1.09,
                1.22,
                1.42,
                1.67,
                1.93,
                2.19,
                2.45,
                2.83,
                3.35,
                3.86,
                4.38,
                4.89,
                5.66,
                6.7,
                7.72,
                8.76,
                9.78,
                11.33,
                13.39,
                15.45,
                17.51,
                19.57,
                22.15,
                25.24,
            ]
        ),
        "mm",
        "Particle diameter of bins",
    )

    spread = common.var_to_dict(
        "spread",
        [
            0.129,
            0.129,
            0.129,
            0.129,
            0.129,
            0.129,
            0.129,
            0.129,
            0.129,
            0.129,
            0.257,
            0.257,
            0.257,
            0.257,
            0.257,
            0.515,
            0.515,
            0.515,
            0.515,
            0.515,
            1.030,
            1.030,
            1.030,
            1.030,
            1.030,
            2.060,
            2.060,
            2.060,
            2.060,
            2.060,
            3.090,
            3.090,
        ],
        "mm",
        "Bin size spread of bins",
    )

    velocity = common.var_to_dict(
        "velocity",
        np.array(
            [
                0.05,
                0.15,
                0.25,
                0.35,
                0.45,
                0.55,
                0.65,
                0.75,
                0.85,
                0.95,
                1.1,
                1.3,
                1.5,
                1.7,
                1.9,
                2.2,
                2.6,
                3,
                3.4,
                3.8,
                4.4,
                5.2,
                6.0,
                6.8,
                7.6,
                8.8,
                10.4,
                12.0,
                13.6,
                15.2,
                17.6,
                20.8,
            ]
        ),
        "m s^-1",
        "Terminal fall velocity for each bin",
    )

    v_spread = [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.4,
        0.4,
        0.4,
        0.4,
        0.4,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        1.6,
        1.6,
        1.6,
        1.6,
        1.6,
        3.2,
        3.2,
    ]

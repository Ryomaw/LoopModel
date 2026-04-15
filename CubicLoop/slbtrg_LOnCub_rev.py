#!/usr/bin/env python3
"""The TRG algorithm for the loop model (general n) on the square lattice rewritten by ryoma
    240524 rewrite the contraction part

Reference:
    M. Levin, C. P. Nave: Phys. Rev. Lett. 99, 120601 (2007)
"""

# import textwrap
import numpy as np

import ising_2d as ising
import common

from ncon import ncon

# import math

import csv
import sys
# pandas / matplotlib are imported lazily in print_graph to reduce startup cost.
# from matplotlib.lines import Line2D
# import itertools

# from rich.progress import Progress
# import time




class BTRG:
    def __init__(self, temp, chi, num, n):
        self.temp = temp
        self.chi = chi
        self.num = num
        self.n = n
        # self.f_exact = ising.exact_free_energy(temp)
        self.f_exact = np.nan

        a, bt, n_spin = common.initial_TN_OnCub_rev_forB(self.temp, self.n)
        self.A = a
        self.S1 = bt
        self.S2 = bt

        

        factor = self.Normalize()
        self.log_factors = [np.log(factor)]
        self.n_spins = [n_spin]
        self.step = 0
        self.target_step = None

    def _results_data_path(self):
        step = self.target_step if self.target_step is not None else self.step
        return f"SLBTRG_LOnCub_rev_results_data_{self.n}_{self.chi}_{step}_{self.num}_{self.temp}.csv"

    def _results_plot_path(self):
        step = self.target_step if self.target_step is not None else self.step
        return f"SLBTRG_LOnCub_rev_results_{self.n}_{self.chi}_{step}_{self.num}_{self.temp}.png"

    @staticmethod
    def _thresholded_power(s, exponent, cutoff=1e-10):
        out = np.zeros_like(s)
        mask = s >= cutoff
        out[mask] = np.power(s[mask], exponent)
        return out

    def _build_apla(self):
        sq_s1 = np.sqrt(self.S1)
        sq_s2 = np.sqrt(self.S2)
        return ncon(
            [self.A, sq_s1, sq_s2, sq_s1, sq_s2],
            [[1, 2, 3, 4], [-1, 1], [-2, 2], [-3, 3], [-4, 4]],
        )

    def Trace(self):
        Apla = self._build_apla()
        trace = np.einsum("ijij->", Apla)
        return trace
    
    def Normalize(self):
        S1_max = np.max(self.S1)
        S2_max = np.max(self.S2)
        self.S1 /= S1_max
        self.S2 /= S2_max

        trace = self.Trace()
        self.A /= trace

        return S1_max * S2_max * trace

    def log_Z(self):
        Apla = self._build_apla()
        trace_a = np.einsum("ijij->", Apla)

        # if trace_a < 0.0:
        #     logging.warning("Negative trace_a %e (%d)", trace_a, self.step)
        log_z = np.sum(np.array(self.log_factors) / np.array(self.n_spins))
        log_z += np.log(abs(trace_a)) / self.n_spins[-1]
        return log_z
    
    def free_energy(self):
        return -self.temp * self.log_Z()

    
    def CalcX(self):
        """Calculate the gauge invariant quantities X1 and X2 from the fixed-point tensor."""
        Apla = self._build_apla()
        
        ta = np.einsum("ijij->", Apla)
        Ta = ta ** 2

        #tb = np.einsum("ijik->jk", self.A)
        #Tb = np.einsum("ij, ji ->", tb, tb)

        #tb = ncon([self.A], [[1, -1, 1, -2]])
        #Tb = ncon([tb, tb], [[1, 2], [2, 1]])

        Tb = ncon([Apla, Apla], [[1, 2, 1, 3], [4, 3, 4, 2]])

        #Tc = np.einsum("ijkl, klij ->", self.A, self.A)

        Tc = ncon([Apla, Apla], [[1, 2, 3, 4], [3, 4, 1, 2]])

        X1 = Ta/Tb
        X2 = Ta/Tc

        return X1, X2
    
    # def Sod(self):
    #     f = self.free_energy()
    #     f_ph = self.free_energy_ph()
    #     f_mh = self.free_energy_mh()
    #     d2fdt2 = (f_ph - 2 * f + f_mh) / (self.h**2)
    #     return d2fdt2
    
    def CentralCharge(self):
        """Calculate the conformal data from the fixed-point tensor."""
        Apla = self._build_apla()
        
        '''
        TraceApla = np.einsum("ijij->", Apla)
        Apla /= TraceApla
        '''
        
        cc = []
    #universal term X の計算
        
        b = ncon([Apla, Apla], [[1, -1, 2, -3], [2, -2, 1, -4]])
        B = ncon([b, b], [[1, 2, 3, 4], [3, 4, 1, 2]])

        X = - np.log(B)/3


        #転送行列 M の作り方2: トーラスを作る
        # M = ncon([Apla], [[-1, 1, -2, 1]])

        M1 = ncon([Apla, Apla], [[-1, 1, -3, 2], [-2, 2, -4, 1]])
        #M = M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])

        M = M1.reshape(
            M1.shape[0] * M1.shape[1],
            M1.shape[2] * M1.shape[3]
            )

        val = np.linalg.eigvals(M)
        val = -np.sort(-val) #descending order

        #central charge
        c = 6*2/np.pi * (np.log(val[0])+2*X)
        #d1 = -1/(2 * np.pi) * (np.log(val[1]) - np.log(val[0]))
        #d2 = -1/(2 * np.pi) * (np.log(val[2]) - np.log(val[0]))

        for i in range(self.num):
            if i == 0:
                cc.append(c.real)       
            elif i < len(val):
                #di = -1 * (2/np.sqrt(3)) * (1/(2 * np.pi)) * np.log(eig[i]/eig[0])
                di = -2 * (1/(2 * np.pi)) * np.log(val[i]/val[0])
                cc.append(di.real)
            else:
                cc.append(None)

        return cc
        
    
    def update(self):
        k = -0.5 #<- the most accurate valur for 2d ising
        #k = 0 #<- normal TRG
        
        # SVD (bottom, left) - (top, right)
        u, s, vt = common.svd(self.A, [2, 3], [0, 1], self.chi)
        #s = np.abs(s)

        TildesF = self._thresholded_power(s, (1 - k) / 2)
        #TildesF = np.array([np.power(s[n], (1-k)/2) for n in range(len(s))])
        TildeMatsF = np.diag(TildesF)

        PowsF = self._thresholded_power(s, k)
        #PowsF = np.array([np.power(s[n], k) for n in range(len(s))])
        MatsF = np.diag(PowsF)

        C = u * TildesF[None, None, :]
        D = vt * TildesF[:, None, None]
        #c3 = u * sqrt_s[None, None, :]
        #c1 = vt * sqrt_s[:, None, None]

        # SVD (top, left) - (right, bottom)
        u, s, vt = common.svd(self.A, [1, 2], [3, 0], self.chi)
        #s = np.abs(s)

        TildesE = self._thresholded_power(s, (1 - k) / 2)
        #TildesF = np.array([np.power(s[n], (1-k)/2) for n in range(len(s))])
        TildeMatsE = np.diag(TildesE)

        PowsE = self._thresholded_power(s, k)
        #PowsF = np.array([np.power(s[n], k) for n in range(len(s))])
        MatsE = np.diag(PowsE)

        #TildesE = np.array([np.power(s[n], (1-k)/2) for n in range(len(s))])
        #TildeMatsE = np.diag(TildesE)
        #PowsE = np.array([np.power(s[n], k) for n in range(len(s))])
        #MatsE = np.diag(PowsE)

        A = u * TildesE[None, None, :]
        B = vt * TildesE[:, None, None]
        #c2 = u * sqrt_s[None, None, :]
        #c0 = vt * sqrt_s[:, None, None]

        # Contraction
        #self.A = np.tensordot(
            #np.tensordot(c0, c1, (1, 2)),
            #np.tensordot(c2, c3, (1, 1)),
            #((1, 3), (2, 0)))
        
        #Ar = ncon([c0, c1], ([-2, 1, -3], [-1, 1, -4]))
        #Al = ncon([c3, c2], ([1, -2, -3], [-1, 1, -4]))
        #self.A = ncon([Ar, Al], ([-1, -2, 1, 2], [2, 1, -3, -4]))

        #modification by Honma-san
        '''
        Ar = ncon([c1, c2], ([-3, 1, -1], [1, -2, -4]))
        Al = ncon([c0, c3], ([-1, -3, 1], [1, -4, -2]))
        self.A = ncon([Ar, Al], ([1, 2, -3, -4], [-2, -1, 1, 2]))
        '''

        L = ncon([B, self.S1, D], [[-1, 1, -3], [1, 2], [ -2, -4, 2]])
        
        R = ncon([C, self.S1, A], [[-1, 2, -3], [1, 2], [ 1, -2, -4]])

        A = ncon([L, self.S2, self.S2, R], [[-1, -2, 3, 4], [1, 3], [2, 4], [1, 2, -4, -3]])

        self.A = A
        self.S1 = MatsE
        self.S2 = MatsF

        factor = self.Normalize()

        self.log_factors.append(np.log(factor))
        self.n_spins.append(2 * self.n_spins[-1])
        
        self.step += 1
        
    



    def print_legend(self):

        '''
        output = f"""\
            # {self.method} for 2d loop gas model on the honeycomb lattice
            # chi= {self.chi}
            # T= {self.temp}
            # 1: step
            # 2: Nspin
            # 3: central charge 
            # 4: 1st scaling dimension
            # 5: 2nd scaling dimension 
            # 6: 3rd 
            # 7: ... """
        print(textwrap.dedent(output))
        '''

        with open(self._results_data_path(), 'a') as df:
            writer = csv.writer(df)
            writer.writerow([f"n = {self.n}, chi = {self.chi}, temp. = {self.temp}"])
            header = [
                "step",
                "Nspin",
                "temp",
                "free_energy",
                "free_energy_exact",
                "free_energy_rel_err",
                "X1",
                "X2",
                "log_factor",
                "central_charge",
            ]
            for i in range(1, self.num):
                header.append(f"scaling_dim_{i}")
            writer.writerow(header)

    def print_results(self):
        Nspin = self.n_spins[-1]
        cc = self.CentralCharge()
        f = self.free_energy()
        # f_err = np.abs(f - self.f_exact) / np.abs(self.f_exact)
        f_err = np.nan

        """
        results = [[f"{self.step:04d}",
                   f"{Nspin:.12e}",
                   f"{f:.12e}"]]
        """
        
        results = []
        results.append(self.step) #0
        results.append(Nspin) #1
        results.append(self.temp) #2
        results.append(f) #3
        results.append(np.nan) #4  # exact free energy unused
        results.append(f_err) #5

        x1, x2 = self.CalcX()
        results.append(x1)
        results.append(x2)
        results.append(self.log_factors[-1])
         
        
        for i in range(len(cc)):
            results.append(cc[i])


        
        with open(self._results_data_path(), 'a') as df:
            writer = csv.writer(df)
            writer.writerow(results)

        

    
    def print_graph(self, show_plot=False):
        import matplotlib
        # Always render to file reliably even in headless terminals.
        matplotlib.use("Agg", force=True)
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(
            self._results_data_path(),
            sep=',', skiprows=2, header=None
            )

        # --- figure & subplots ---
        fig, axes = plt.subplots(
            1, 2, figsize=(8, 10), sharex=True
        )

        # =========================
        # 左:central charge & scaling dimensions
        # =========================
        ax0 = axes[0]

        cc_col = 9
        x1_col = 6
        x2_col = 7

        # central charge
        ax0.scatter(df[0], df[cc_col], marker='x', color = 'red', label='central charge')
        ax0.hlines(1.0, 0, self.step, linestyles='dotted', color='black')

        # scaling dimensions
        for i in range(cc_col + 1, cc_col + self.num):
            ax0.scatter(df[0], df[i], marker='.')

        ax0.hlines(1/6, 0, self.step, linestyles='dotted', color='black')
        ax0.hlines(2/3,   0, self.step, linestyles='dotted', color='black')
        ax0.hlines(3/2, 0, self.step, linestyles = 'dotted', color = 'black')

        ax0.set_ylim(0, 1.5)

        ax0.set_ylabel('scaling dimension / c')
        ax0.legend()
        ax0.set_title(
            f"""central charge and scaling dimensions of
            loop model (n={self.n}) on square lattice by BTRG
            χ = {self.chi}, RG step = {self.step}, T = {self.temp}"""
        )

        # =========================
        # 右:relative error of free energy
        # =========================
        ax1 = axes[1]

        ax1.scatter(
            df[0], df[x1_col],
            marker='.',
            label='$X_1$'
        )
        ax1.scatter(df[0], df[x2_col], marker = '.', label = '$X_2$')

        #ax1.set_yscale('log')  # 相対誤差なので log 表示を推奨
        ax1.set_xlabel('RG step')
        ax1.set_ylabel('relative error')
        ax1.legend()
        ax1.grid(True, which='both')

        # --- save & show ---
        plt.tight_layout()
        plt.savefig(self._results_plot_path(), dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(fig)

    @staticmethod
    def _print_progress(current, total, width=30):
        if total <= 0:
            return
        ratio = current / total
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)")
        sys.stdout.flush()

    

    
    
    def run(self, step):
        self.target_step = step

        self.print_legend()
        self.print_results()
        self._print_progress(0, step)
        for i in range(step):
            self.update()
            self.print_results()
            self._print_progress(i + 1, step)
        print()
        print('Calculating done!')
        self.print_results()
        self.print_graph(show_plot=False)


def run_temperature_scan(chi, step, num, n, t_min, t_max, t_count, include_tc_point=True, show_plot=False):
    import pandas as pd
    import matplotlib.pyplot as plt

    ts = list(np.linspace(t_min, t_max, t_count))
    ts = sorted(ts)

    x1_values = []
    x2_values = []
    conformal_values = []

    for idx, temp in enumerate(ts, start=1):
        print(f"\nTemperature sweep {idx}/{len(ts)}: T = {temp}")
        btrg = BTRG(temp, chi, num, n)
        btrg.target_step = step

        with open(btrg._results_data_path(), "w"):
            pass

        btrg.run(step)

        # Read X1/X2 from the new per-temperature CSV format.
        df = pd.read_csv(btrg._results_data_path(), sep=",", skiprows=1, header=None)
        x1_values.append(float(df[6].iloc[-1]))
        x2_values.append(float(df[7].iloc[-1]))
        conformal_values.append([float(df[i].iloc[-1]) for i in range(9, 9 + num)])

    summary_data = {"T": ts, "X1": x1_values, "X2": x2_values, "central_charge": [row[0] for row in conformal_values]}
    for i in range(1, num):
        summary_data[f"scaling_dim_{i}"] = [row[i] for row in conformal_values]
    summary = pd.DataFrame(summary_data)
    out_csv = (
        f"SLBTRG_LOnCub_rev_X12_vs_T_data_{n}_{chi}_{step}_{num}_{t_min}_{t_max}_{t_count}.csv"
    )
    out_png = f"SLBTRG_LOnCub_rev_X12_vs_T_{n}_{chi}_{step}_{num}_{t_min}_{t_max}_{t_count}.png"
    out_cc_png = f"SLBTRG_LOnCub_rev_conformal_data_{n}_{chi}_{step}_{num}_{t_min}_{t_max}_{t_count}.png"
    summary.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(ts, x1_values, marker="o", label="$X_1$")
    ax.plot(ts, x2_values, marker="^", label="$X_2$")
    ax.set_xlabel("x")
    ax.set_ylabel("X")
    ax.set_title(f"X1 and X2 vs x (n={n}, chi={chi}, step={step}, num={num})")
    ax.grid(True, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)

    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 6))
    ax2.plot(ts, [row[0] for row in conformal_values], marker="x", label="central charge")
    for i in range(1, num):
        ax2.plot(ts, [row[i] for row in conformal_values], marker=".", label=f"$\u0394_{i}$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("conformal data")
    ax2.set_title(f"Conformal data vs x (n={n}, chi={chi}, step={step}, num={num})")
    ax2.grid(True, which="both")
    ax2.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_cc_png, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig2)

    print(f"Saved temperature-scan summary CSV: {out_csv}")
    print(f"Saved temperature-scan plot PNG: {out_png}")
    print(f"Saved conformal-data plot PNG: {out_cc_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BTRG simulation of the loop model (general n) on square lattice",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("n", type=int, default=2, nargs="?", help="Loop model parameter n")
    parser.add_argument("chi", type=int, default=16, nargs="?", help="Bond dimension")
    parser.add_argument("step", type=int, default=16, nargs="?", help="TRG steps")
    parser.add_argument("num", type=int, default=16, nargs="?", help="the number of eigs of TM")    
    parser.add_argument("T", type=float, default=1.0, nargs="?", help="Temperature")
    parser.add_argument("--show-plot", action="store_true", help="Display plot window in addition to saving PNG")
    parser.add_argument("--scan-t", action="store_true", help="Run RG for each temperature and plot X1/X2 vs T")
    parser.add_argument("--t-min", type=float, default=0.1, help="Minimum temperature for --scan-t")
    parser.add_argument("--t-max", type=float, default=5.0, help="Maximum temperature for --scan-t")
    parser.add_argument("--t-count", type=int, default=50, help="Number of temperature points for --scan-t")
    parser.add_argument("--show-scan-plot", action="store_true", help="Display X1/X2-vs-T plot window in --scan-t")
    args = parser.parse_args()

    N = args.n
    Chi = args.chi
    Step = args.step
    Num = args.num
    T = args.T
    ShowPlot = args.show_plot

    btrg = BTRG(T, Chi, Num, N)
    btrg.target_step = Step

    with open(btrg._results_data_path(), 'w'):
        pass

    if args.scan_t:
        run_temperature_scan(
            chi=Chi,
            step=Step,
            num=Num,
            n=N,
            t_min=args.t_min,
            t_max=args.t_max,
            t_count=args.t_count,
            include_tc_point=False,
            show_plot=args.show_scan_plot,
        )
    else:
        btrg.run(Step)
        if ShowPlot:
            btrg.print_graph(show_plot=True)

r'''
xmin = 0.1
xmax = 5.0

Ts = np.linspace(xmin, xmax, 50)


for i in range(len(Ts)):
    BTRG(Ts[i], Chi, Num, N).run(Step)

if T != common.Tc_O2Cub:
    BTRG(common.Tc_O2Cub, Chi, Num, N).run(Step)




with Progress() as progress:
    task = progress.add_task("[blue]Calculating...", total=len(Ts) + 1 if T != ising.T_C else len(Ts))
    for i in range(len(Ts)):
        BTRG(Ts[i], Chi, Num, N).run(Step)
        progress.update(task, advance=1)
'''

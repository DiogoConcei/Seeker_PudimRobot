import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class BenchmarkPlotter:
    def __init__(self, csv_path="data/benchmark.csv"):
        self.csv_path = csv_path
        self.output_dir = "plots"

        # Cria a pasta de plots se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path, sep=';')
            # Filtra apenas linhas onde houve inferência (ignora o sleep/zeros)
            # Isso é crucial para medir a performance do HARDWARE
            df_active = df[df['infer_ms'] > 0].copy()
            return df, df_active
        except FileNotFoundError:
            print("❌ Arquivo CSV não encontrado!")
            return None, None

    def generate_all(self):
        df, df_active = self.load_data()
        if df is None or df.empty: return

        sns.set_theme(style="darkgrid")

        print("🎨 Gerando gráficos...")
        self.plot_stability(df_active)
        self.plot_thermal_throttling(df_active)
        self.plot_fps_comparison(df_active)
        print(f"✅ Gráficos salvos na pasta '{self.output_dir}'")

    def plot_stability(self, df):
        """
        - Caixa pequena = Performance consistente (Bom).
        - Caixa grande = Performance instável (Ruim).
        - Pontos fora (outliers) = Travadas aleatórias.
        """
        plt.figure(figsize=(12, 6))

        # Cria o gráfico
        sns.boxplot(x='mode', y='infer_ms', hue='infra', data=df, palette="viridis")

        plt.title('Estabilidade de Latência (Menor é Melhor)', fontsize=16)
        plt.ylabel('Tempo de Inferência (ms)')
        plt.xlabel('Modo de Operação')
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/1_estabilidade_latency.png")
        plt.close()

    def plot_thermal_throttling(self, df):
        """
        Gráfico de Linha: (Aquecimento)
        - Linha subindo = O hardware está esquentando e ficando lento.
        - Linha reta = O sistema está refrigerado e saudável.
        """
        plt.figure(figsize=(12, 6))

        # Plotamos o tempo de inferência ao longo dos frames
        sns.lineplot(x='frame', y='infer_ms', hue='mode', style='infra', data=df, alpha=0.8)

        plt.title('Evolução Térmica: Latência ao longo do tempo', fontsize=16)
        plt.ylabel('Tempo de Inferência (ms)')
        plt.xlabel('Número do Frame (Tempo decorrido)')
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/2_thermal_throttling.png")
        plt.close()

    def plot_fps_comparison(self, df):
        """
        Gráfico de Barras: Potencial do Hardware
        Mostra a força bruta média de cada modo/infra.
        """
        plt.figure(figsize=(10, 6))

        # Calcula a média de HW_FPS por modo e infra
        summary = df.groupby(['mode', 'infra'])['hw_fps'].mean().reset_index()

        sns.barplot(x='mode', y='hw_fps', hue='infra', data=summary, palette="magma")

        plt.title('Potencial Bruto do Hardware (FPS "Justo")', fontsize=16)
        plt.ylabel('FPS Médio (Sem contar sleep)')
        plt.xlabel('Modo')
        plt.tight_layout()

        plt.savefig(f"{self.output_dir}/3_hardware_potential.png")
        plt.close()
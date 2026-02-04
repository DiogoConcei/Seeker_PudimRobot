from watcher import Watcher
from plotter import BenchmarkPlotter

if __name__ == "__main__":
    observador = Watcher(mode=2,show_display=True,use_threading=True)
    observador.start()

    # print("Gerando relatórios visuais...")
    plotter = BenchmarkPlotter()
    plotter.generate_all()
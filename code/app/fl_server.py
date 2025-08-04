import time
from pathlib import Path

from flwr.common import Parameters, Scalar
from flwr.common.typing import FitIns
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.server import ClientProxy, FitResultsAndFailures, fit_clients
from flwr.server.strategy import Strategy


def print_fn(msg: str, fn_name: str = "fl-server"):
    """Prints messages with a prefix."""
    print(f"[FLServer] ({fn_name}) {msg}")


class FLServer(Server):
    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Strategy | None = None,
        communication_file_path: Path | None = None,
        fit_round_file_path: Path | None = None,
        print_fn=print_fn,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.communication_file_path = communication_file_path
        self.fit_round_file_path = fit_round_file_path
        self.print_fn = print_fn
        print_fn("Initialized with custom server class.", "fl-server.init")

    def fit_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[Parameters | None, dict[str, Scalar], FitResultsAndFailures] | None:
        """Perform a single round of federated averaging."""

        def print_(msg: str):
            self.print_fn(msg, "fl-server.fit_round")

        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            print_("configure_fit: no clients selected, cancel")
            return None
        print_(
            f"configure_fit: strategy sampled {len(client_instructions)} clients "
            f"(out of {self._client_manager.num_available()})",
        )

        client_instructions_comm: list[tuple[ClientProxy, FitIns]] = []

        for client_proxy, ins in client_instructions:
            ins_comm = FitIns(
                parameters=ins.parameters,
                config=dict(**ins.config, only_communication=True),
            )
            client_instructions_comm.append((client_proxy, ins_comm))

        # Communication time measurement
        print_(
            f"Communication measurement for round {server_round} with "
            f"{len(client_instructions_comm)} clients.",
        )
        start_time_comm = time.perf_counter()

        results, failures = fit_clients(
            client_instructions=client_instructions_comm,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )

        end_time_comm = time.perf_counter()
        print_(
            f"Communication time for round {server_round}: "
            f"{end_time_comm - start_time_comm:.6f} seconds.",
        )

        if self.communication_file_path is not None:
            print_(f"Writing communication time to {self.communication_file_path}")

            with open(self.communication_file_path, "a") as f:
                f.write(
                    f"Communication time for round {server_round}: "
                    f"{end_time_comm - start_time_comm:.6f} seconds\n"
                )

        print_(f"Running fit round {server_round} ...")
        start_time = time.perf_counter()

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )

        end_time = time.perf_counter()
        print_(
            f"Fit round {server_round} completed in "
            f"{end_time - start_time:.6f} seconds."
        )

        if self.fit_round_file_path is not None:
            print_(f"Writing fit round time to {self.fit_round_file_path}")
            with open(self.fit_round_file_path, "a") as f:
                f.write(
                    f"Fit round {server_round} completed in "
                    f"{end_time - start_time:.6f} seconds\n"
                )

        print_(
            f"aggregate_fit: received "
            f"{len(results)} results and {len(failures)} failures"
        )

        # Aggregate training results
        aggregated_result: tuple[
            Parameters | None,
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

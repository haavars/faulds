defmodule Faulds.Logistic do
  @moduledoc """
  L2-penalised logistic regression by Newton's method, on standardised
  features, for the fusion and calibration (`LearnedFusion`,
  `SearchCalibration`). The rows come in chunks, each `{x, y}` with `x`
  `{n, features}` and `y` `{n}` (1 or 0), so a fit over millions of pairs
  never builds one matrix of them.
  """

  @doc """
  Fits on `chunks`. Options: `:l2` (default 1.0, on standardised
  features; the bias is not penalised), `:iterations` (Newton steps,
  default 20), `:balance` (default true: the two classes weigh the same
  in total; false: every row weighs 1, so the output is a probability
  under the training rows' own mix).

  `%{mean:, std:, weights:, bias:, rows:, positives:}`, the weights on
  `(x − mean) / std`.
  """
  @spec fit([{Nx.Tensor.t(), Nx.Tensor.t()}], keyword()) :: map()
  def fit(chunks, opts \\ []) do
    l2 = Keyword.get(opts, :l2, 1.0)
    iterations = Keyword.get(opts, :iterations, 20)
    balance = Keyword.get(opts, :balance, true)

    data = Enum.map(chunks, fn {x, y} -> {Nx.as_type(x, :f64), Nx.as_type(y, :f64)} end)
    {_, f} = data |> hd() |> elem(0) |> Nx.shape()

    n = data |> Enum.map(fn {x, _} -> elem(Nx.shape(x), 0) end) |> Enum.sum()
    positives = data |> Enum.map(fn {_, y} -> y |> Nx.sum() |> Nx.to_number() end) |> Enum.sum()

    # Standardise over all rows.
    sum = data |> Enum.map(fn {x, _} -> Nx.sum(x, axes: [0]) end) |> Enum.reduce(&Nx.add/2)
    mean = Nx.divide(sum, n)

    var =
      data
      |> Enum.map(fn {x, _} -> x |> Nx.subtract(mean) |> Nx.pow(2) |> Nx.sum(axes: [0]) end)
      |> Enum.reduce(&Nx.add/2)
      |> Nx.divide(n)

    std = var |> Nx.sqrt() |> Nx.max(1.0e-9)

    data =
      Enum.map(data, fn {x, y} ->
        rows = elem(Nx.shape(x), 0)

        xs =
          Nx.concatenate(
            [
              Nx.divide(Nx.subtract(x, mean), std),
              Nx.broadcast(Nx.tensor(1.0, type: :f64), {rows, 1})
            ],
            axis: 1
          )

        # Balanced: each class weighs as much as the other, over all n.
        w_row =
          if balance,
            do: Nx.select(Nx.equal(y, 1.0), n / (2 * positives), n / (2 * (n - positives))),
            else: Nx.broadcast(Nx.tensor(1.0, type: :f64), {rows})

        {xs, y, w_row}
      end)

    penalty =
      Nx.eye(f + 1, type: :f64)
      |> Nx.multiply(l2)
      |> Nx.put_slice([f, f], Nx.tensor([[0.0]], type: :f64))

    beta =
      Enum.reduce(1..iterations, Nx.broadcast(Nx.tensor(0.0, type: :f64), {f + 1}), fn _, beta ->
        {grad, hessian} =
          data
          |> Enum.map(fn {xs, y, w_row} ->
            p = xs |> Nx.dot(beta) |> Nx.sigmoid()
            w = Nx.multiply(w_row, Nx.multiply(p, Nx.subtract(1.0, p)))

            {Nx.dot(Nx.transpose(xs), Nx.multiply(w_row, Nx.subtract(p, y))),
             Nx.dot(Nx.transpose(Nx.multiply(xs, Nx.new_axis(w, 1))), xs)}
          end)
          |> Enum.reduce(fn {g, h}, {ga, ha} -> {Nx.add(g, ga), Nx.add(h, ha)} end)

        grad = Nx.add(grad, Nx.dot(penalty, beta))
        Nx.subtract(beta, Nx.LinAlg.solve(Nx.add(hessian, penalty), grad))
      end)

    beta = Nx.to_flat_list(beta)

    %{
      mean: Nx.to_flat_list(mean),
      std: Nx.to_flat_list(std),
      weights: Enum.take(beta, f),
      bias: List.last(beta),
      rows: n,
      positives: round(positives)
    }
  end

  @doc """
  The linear score `w · (x − mean) / std + b` folded into `{scale,
  offset}`, so a caller dots raw features with `scale` and adds `offset`.
  """
  @spec folded(map()) :: {[float()], float()}
  def folded(%{weights: weights, std: std, mean: mean, bias: bias}) do
    scale = Enum.zip_with(weights, std, &(&1 / &2))
    {scale, bias - (Enum.zip_with(scale, mean, &(&1 * &2)) |> Enum.sum())}
  end
end

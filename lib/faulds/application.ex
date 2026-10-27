defmodule Faulds.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Matchers that keep a model loaded run as processes under this.
      {DynamicSupervisor, name: Faulds.MatcherSupervisor, strategy: :one_for_one}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Faulds.Supervisor)
  end
end

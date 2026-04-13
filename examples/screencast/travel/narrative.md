# Get to the Wedding

Your best friend is getting married. Ceremony is at 3pm Saturday, in a city
you don't live in. You need to get there on time.

This is a planning problem. You have options — flights, trains, rental cars —
each with different costs and constraints. A good agent should pick the
cheapest option, and when that option falls through, find the next best one
without you having to spell out every contingency in advance.

## The traditional approach

In a standard LangGraph workflow, you wire each fallback by hand. Search
flights. If flights are available, book a direct. If direct is cancelled, try
connecting. If all flights are grounded, search ground transport. If the train
is sold out, book a rental car. Every branch is an explicit conditional edge,
every fallback is a routing function you wrote and maintain.

It works. But the routing logic scales with the number of options times the
number of failure modes. Three transport modes, four routing functions, four
conditional edges. Add a bus option? Edit the routing functions, update the
type hints, add more edges. The graph structure — not just the business
logic — has to change every time.

```
before.py — 4 routing functions, 4 conditional edges
```

## The GOAP approach

With GOAP, you describe the same options as actions with preconditions,
effects, and costs. `search_flights` requires `has_destination`, produces
`flight_options_found`, costs 1.0. `book_direct_flight` requires
`flight_options_found`, produces `travel_booked`, costs 2.0. And so on.

You state the goal: `{at_venue: True, on_time: True}`.

The planner finds the cheapest sequence of actions that reaches the goal from
the current world state. On a clean run, that's: search flights, book
direct, book hotel, confirm arrival. Total cost 5. No routing code.

```
after.py — 0 routing functions, 0 conditional edges
```

## When things go wrong

The direct flight gets cancelled. The execute function raises an exception.
The observer detects the failure, blacklists that action, and sends the
state back to the planner. The planner re-searches from the current world
state — flights have already been searched, so it picks the next cheapest
booking option: connecting flight at cost 3. The rest of the chain
continues. Goal achieved.

Then the weather grounds all flights. Now `search_flights` itself fails.
The planner has no flight options at all, so it routes through a completely
different branch: `search_ground_transport` then `book_train`. Different
precondition chain, same goal. No code changed.

Then the train sells out too. The planner falls through to `book_rental_car`
— the most expensive ground option, but the only one left that reaches the
goal. Two replans, three failed actions, still arrives at the venue by 1pm.

```
after_disrupted.py — same actions, three cascading failures, all recovered
```

## What this demonstrates

The point is not that traditional workflows can't handle failures — they
can, if you wire every fallback. The point is that GOAP separates the
problem description (what actions exist, what they require, what they
produce) from the solution strategy (which actions to run, in what order).

When you add a new option — say, a bus route — you add one `ActionSpec`.
The planner immediately considers it alongside existing options, slotted
into the right place by cost, without any changes to graph structure or
routing logic.

When an option fails at runtime, the planner adapts by re-searching the
action space. You don't need to anticipate which specific failures to
recover from. If there exists any sequence of remaining actions that reaches
the goal, the planner will find it.

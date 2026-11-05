-- Long-Horizon Trip Agent schema. All environment data is synthetic.
--
-- Access model: row-level security is enabled on every table with no policies, so the public
-- anon key can read nothing. The experiment runner uses the secret (service-role) key, which
-- bypasses RLS. The agent itself never queries the database: it only sees tool results.

-- Datasets: a trusted catalog (places + travel legs) and scenarios with hidden reference itineraries.
create table if not exists public.datasets (
  name text primary key,
  catalog_meta jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.places (
  dataset text not null references public.datasets(name) on delete cascade,
  id text not null,
  kind text not null check (kind in ('attraction', 'restaurant', 'hotel')),
  name text not null,
  neighborhood text not null,
  data jsonb not null,
  primary key (dataset, id)
);

create table if not exists public.travel_legs (
  dataset text not null,
  from_id text not null,
  to_id text not null,
  minutes integer not null check (minutes >= 0),
  cost_per_person double precision not null default 0,
  primary key (dataset, from_id, to_id),
  foreign key (dataset, from_id) references public.places(dataset, id) on delete cascade,
  foreign key (dataset, to_id) references public.places(dataset, id) on delete cascade
);

create table if not exists public.scenarios (
  dataset text not null references public.datasets(name) on delete cascade,
  id text not null,
  split text not null check (split in ('dev', 'eval')),
  horizon integer not null check (horizon > 0),
  data jsonb not null,
  primary key (dataset, id)
);

-- Experiments, runs and scores.
create table if not exists public.experiments (
  name text primary key,
  dataset text not null,
  dataset_hash text not null,
  config jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists public.runs (
  run_key text primary key,
  experiment text not null references public.experiments(name) on delete cascade,
  scenario_id text not null,
  split text not null,
  horizon integer not null,
  condition text not null check (condition in ('clean', 'corrupted')),
  variant text not null,
  repeat integer not null,
  seed bigint not null,
  model text not null,
  status text not null check (status in ('completed', 'error')),
  stop_reason text,
  tool_calls integer,
  verify_calls integer,
  turns integer,
  prompt_tokens integer,
  completion_tokens integer,
  wall_ms double precision,
  record jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists runs_experiment_idx on public.runs (experiment);

create table if not exists public.run_events (
  run_key text not null references public.runs(run_key) on delete cascade,
  seq integer not null,
  type text not null,
  data jsonb not null,
  primary key (run_key, seq)
);

create table if not exists public.run_scores (
  run_key text primary key references public.runs(run_key) on delete cascade,
  experiment text not null references public.experiments(name) on delete cascade,
  valid boolean not null,
  fraction_satisfied double precision not null,
  false_reliance_items integer not null,
  first_error_position integer,
  recovery_success_rate double precision,
  score jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists run_scores_experiment_idx on public.run_scores (experiment);

-- One row per run with its headline scores, for the Supabase table editor or SQL analysis.
create or replace view public.run_results with (security_invoker = true) as
select
  r.experiment, r.run_key, r.scenario_id, r.split, r.horizon, r.condition, r.variant, r.repeat, r.model,
  r.status, r.stop_reason, r.tool_calls, r.verify_calls, r.turns, r.prompt_tokens, r.completion_tokens, r.wall_ms,
  s.valid, s.fraction_satisfied, s.false_reliance_items, s.first_error_position, s.recovery_success_rate
from public.runs r
left join public.run_scores s using (run_key);

alter table public.datasets enable row level security;
alter table public.places enable row level security;
alter table public.travel_legs enable row level security;
alter table public.scenarios enable row level security;
alter table public.experiments enable row level security;
alter table public.runs enable row level security;
alter table public.run_events enable row level security;
alter table public.run_scores enable row level security;

grant select, insert, update, delete on all tables in schema public to service_role;

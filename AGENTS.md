<!-- SKILL_ALIASES_BLOCK_START -->
## Skills
A skill is a set of local instructions stored in `SKILL.md`.

### Available skills
- quant-hypothesis-lab: Generate, refine, and stress-test trading strategy hypotheses for crypto with falsifiable specs, regime map, data contract, and experiment plan. Activation word: `Idea`. (file: /Users/hvost/Documents/Algotrading/Profitable Strategies/Backtests/quant-hypothesis-lab/SKILL.md)
- quant-backtest-builder: Implement backtest-ready Python code from strategy specs with cost model, reproducibility, and artifacts. Activation word: `Backtest`. (file: /Users/hvost/Documents/Algotrading/Profitable Strategies/Backtests/quant-backtest-builder/SKILL.md)
- quant-robustness-optimizer: Run walk-forward, regime splits, stress tests, and constrained optimization for robust parameter selection. Activation word: `Optimization`. (file: /Users/hvost/Documents/Algotrading/Profitable Strategies/Backtests/quant-robustness-optimizer/SKILL.md)
- quant-iteration-engine: Diagnose performance failures, propose ranked improvements, and run controlled one-change iterations with ablation evidence. Activation word: `BacktestImprove`. (file: /Users/hvost/Documents/Algotrading/Profitable Strategies/Backtests/quant-iteration-engine/SKILL.md)
- exec-cex-production-engine: Build production Python execution engine for Binance and Bitget with OMS, reconciliation, risk gates, and observability. Activation word: `Live`. (file: /Users/hvost/Documents/Algotrading/Profitable Strategies/Backtests/exec-cex-production-engine/SKILL.md)
- ops-live-feedback-loop: Analyze live trading telemetry, detect drift, quantify execution drag, and propose safe rollout improvements. Activation word: `LiveImprove`. (file: /Users/hvost/Documents/Algotrading/Profitable Strategies/Backtests/ops-live-feedback-loop/SKILL.md)

### Trigger policy (strict)
- Activate these skills only on explicit user call.
- Accept explicit calls via activation words: `Idea`, `Backtest`, `Optimization`, `BacktestImprove`, `Live`, `LiveImprove`.
- Accept explicit calls via skill tokens: `$quant-hypothesis-lab`, `$quant-backtest-builder`, `$quant-robustness-optimizer`, `$quant-iteration-engine`, `$exec-cex-production-engine`, `$ops-live-feedback-loop`.
- Do not auto-trigger by topic similarity.
<!-- SKILL_ALIASES_BLOCK_END -->

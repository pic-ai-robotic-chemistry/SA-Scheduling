from gymnasium.envs.registration import register

# Disable wrappers for direct state access and the environment's custom return values.
# order_enforce=False: Skip OrderEnforcing to allow direct access to custom attributes such as env.state

register(
    id="fjsp-v0",  # Environment name (including version number)
    entry_point="rl.env.fjsp_env:FJSPEnv",  # The location of the environment class, like 'foldername.filename:classname'
    order_enforce=False,
    disable_env_checker=True,
)

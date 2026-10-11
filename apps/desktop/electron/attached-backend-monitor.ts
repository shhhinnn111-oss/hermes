import { isHostResourcePressureError } from './backend-health'
import { isAttachedBackendTokenDrifted } from './dashboard-token'

export interface AttachedBackendMonitorOptions {
  probe: () => Promise<void>
  readToken: () => Promise<string>
  adoptedToken: string
  onGone: () => void
}

/** One liveness probe at a time; inability to probe never authorizes respawn. */
export function startAttachedBackendMonitorLoop(options: AttachedBackendMonitorOptions, pollMs = 15_000): () => void {
  let stopped = false
  let probing = false

  const stop = () => {
    stopped = true
    clearInterval(timer)
  }

  const timer = setInterval(async () => {
    if (stopped || probing) {
      return
    }

    probing = true

    try {
      await options.probe()

      if (stopped) {
        return
      }

      const servedToken = await options.readToken().catch(() => options.adoptedToken)

      if (isAttachedBackendTokenDrifted({ servedToken, adoptedToken: options.adoptedToken })) {
        throw new Error('attached backend is serving a different session token')
      }
    } catch (error) {
      if (!stopped && !isHostResourcePressureError(error)) {
        stop()
        options.onGone()
      }
    } finally {
      probing = false
    }
  }, pollMs)

  timer.unref?.()

  return stop
}

import { expect, test, vi } from 'vitest'

import { startAttachedBackendMonitorLoop } from './attached-backend-monitor'
import { waitForHermesReady } from './backend-health'

test('resource pressure waits for the next monitor tick; a recovered host stays attached and a dead port recovers', async () => {
  vi.useFakeTimers()
  let code: string | null = 'ENOBUFS'
  const onGone = vi.fn()

  const fetchPublicJson = vi.fn(async () => {
    if (code) {
      throw Object.assign(new Error(`connect ${code}`), { code })
    }

    return { ok: true }
  })

  const stop = startAttachedBackendMonitorLoop({
    adoptedToken: 'session-token',
    probe: () =>
      waitForHermesReady('http://127.0.0.1:2802', {
        alreadyBound: true,
        fetchPublicJson,
        fetchJson: async () => ({})
      }),
    readToken: async () => 'session-token',
    onGone
  })

  try {
    await vi.advanceTimersByTimeAsync(15_000)
    expect(fetchPublicJson).toHaveBeenCalledTimes(1)
    expect(onGone).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(14_999)
    expect(fetchPublicJson).toHaveBeenCalledTimes(1)
    code = null
    await vi.advanceTimersByTimeAsync(1)
    expect(fetchPublicJson).toHaveBeenCalledTimes(2)
    expect(onGone).not.toHaveBeenCalled()
    code = 'ECONNREFUSED'
    await vi.advanceTimersByTimeAsync(15_000)
    expect(onGone).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(fetchPublicJson).toHaveBeenCalledTimes(3)
  } finally {
    stop()
    vi.useRealTimers()
  }
})

test('slow monitor probes never overlap and a stopped monitor cannot recover a superseded attachment', async () => {
  vi.useFakeTimers()
  let rejectProbe!: (error: Error) => void

  const pending = new Promise<void>((_resolve, reject) => {
    rejectProbe = reject
  })

  const probe = vi.fn(() => pending)
  const onGone = vi.fn()
  const readToken = vi.fn(async () => 'session-token')
  const stop = startAttachedBackendMonitorLoop({ adoptedToken: 'session-token', probe, readToken, onGone })

  try {
    await vi.advanceTimersByTimeAsync(180_000)
    expect(probe).toHaveBeenCalledTimes(1)
    stop()
    rejectProbe(new Error('old backend died after attachment was replaced'))
    await vi.advanceTimersByTimeAsync(15_000)
    expect(onGone).not.toHaveBeenCalled()
    expect(readToken).not.toHaveBeenCalled()
  } finally {
    stop()
    vi.useRealTimers()
  }
})

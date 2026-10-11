import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesGateway } from '@/hermes'

import {
  MCP_RELOAD_BUSY_RETRIES,
  MCP_RELOAD_TIMEOUT_MS,
  mcpReloadSlashParams,
  requestMcpReload,
  requestMcpReloadWhenIdle
} from './mcp-reload'

function gatewayWith(response: unknown) {
  const request = vi.fn().mockResolvedValue(response)

  return { gateway: { request } as unknown as HermesGateway, request }
}

describe('requestMcpReload', () => {
  it('defaults an omitted failed list to empty for older backends', async () => {
    const { gateway } = gatewayWith({ status: 'reloaded' })

    await expect(requestMcpReload(gateway)).resolves.toEqual({
      failed: [],
      message: null,
      status: 'reloaded'
    })
  })

  it('keeps only string entries in failed', async () => {
    const { gateway } = gatewayWith({ failed: ['broken', 3, null, 'flaky'], status: 'reloaded' })

    await expect(requestMcpReload(gateway)).resolves.toEqual(expect.objectContaining({ failed: ['broken', 'flaky'] }))
  })

  it('reports the confirm_required verdict with its backend message', async () => {
    const { gateway } = gatewayWith({ message: 'reply now to proceed', status: 'confirm_required' })

    await expect(requestMcpReload(gateway)).resolves.toEqual(
      expect.objectContaining({ message: 'reply now to proceed', status: 'confirm_required' })
    )
  })

  it('maps a missing status to reloaded', async () => {
    const { gateway } = gatewayWith({})

    await expect(requestMcpReload(gateway)).resolves.toEqual(expect.objectContaining({ status: 'reloaded' }))
  })

  it('throws on an unexpected status value', async () => {
    const { gateway } = gatewayWith({ status: 'queued' })

    await expect(requestMcpReload(gateway)).rejects.toThrow(/unexpected reload.mcp status: queued/)
  })

  it('sends the resolved profile name for an explicit (connection, profile) scope', async () => {
    const { gateway, request } = gatewayWith({ status: 'reloaded' })

    await requestMcpReload(gateway, { confirm: true, profile: { connectionId: 'local', profile: 'work' } })

    expect(request).toHaveBeenCalledWith('reload.mcp', { confirm: true, profile: 'work' }, MCP_RELOAD_TIMEOUT_MS)
  })

  it('omits confirm, always and session_id when they are not requested', async () => {
    const { gateway, request } = gatewayWith({ status: 'reloaded' })

    await requestMcpReload(gateway, { sessionId: null })

    expect(request).toHaveBeenCalledWith('reload.mcp', {}, MCP_RELOAD_TIMEOUT_MS)
  })

  it('forwards confirm, always and the session id when given', async () => {
    const { gateway, request } = gatewayWith({ status: 'reloaded' })

    await requestMcpReload(gateway, { always: true, confirm: true, sessionId: 's-1' })

    expect(request).toHaveBeenCalledWith(
      'reload.mcp',
      { always: true, confirm: true, session_id: 's-1' },
      MCP_RELOAD_TIMEOUT_MS
    )
  })

  it('forwards the session id; the backend uses it only for a chat of that profile', async () => {
    const { gateway, request } = gatewayWith({ status: 'reloaded' })

    await requestMcpReload(gateway, { profile: 'work', sessionId: 's-any' })

    expect(request).toHaveBeenCalledWith('reload.mcp', { profile: 'work', session_id: 's-any' }, MCP_RELOAD_TIMEOUT_MS)
  })

  describe('requestMcpReloadWhenIdle', () => {
    const busy = () => Object.assign(new Error('session busy — a turn is running in s-1'), { code: 4009 })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('retries while a turn is running, then reloads', async () => {
      vi.useFakeTimers()
      const request = vi.fn().mockRejectedValueOnce(busy()).mockResolvedValueOnce({ status: 'reloaded' })
      const pending = requestMcpReloadWhenIdle({ request } as unknown as HermesGateway, { confirm: true })

      await vi.runAllTimersAsync()

      await expect(pending).resolves.toEqual(expect.objectContaining({ status: 'reloaded' }))
      expect(request).toHaveBeenCalledTimes(2)
    })

    it('gives up after a bounded number of tries and reports the refusal', async () => {
      vi.useFakeTimers()
      const request = vi.fn().mockRejectedValue(busy())
      const pending = requestMcpReloadWhenIdle({ request } as unknown as HermesGateway, { confirm: true })
      const settled = expect(pending).rejects.toThrow(/session busy/)

      await vi.runAllTimersAsync()

      await settled
      expect(request.mock.calls.length).toBeGreaterThan(1)
      expect(request.mock.calls.length).toBeLessThanOrEqual(MCP_RELOAD_BUSY_RETRIES)
    })

    it('does not retry any other failure', async () => {
      const request = vi.fn().mockRejectedValue(new Error('server down'))

      await expect(requestMcpReloadWhenIdle({ request } as unknown as HermesGateway)).rejects.toThrow('server down')
      expect(request).toHaveBeenCalledTimes(1)
    })
  })

  it('retries once without profile when an older backend rejects it', async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error('invalid params for reload.mcp: profile: unexpected key'))
      .mockResolvedValueOnce({ status: 'reloaded' })

    const gateway = { request } as unknown as HermesGateway

    await expect(requestMcpReload(gateway, { always: true, confirm: true, profile: 'work' })).resolves.toEqual({
      failed: [],
      message: null,
      status: 'reloaded'
    })

    expect(request).toHaveBeenNthCalledWith(
      1,
      'reload.mcp',
      { always: true, confirm: true, profile: 'work' },
      MCP_RELOAD_TIMEOUT_MS
    )
    expect(request).toHaveBeenNthCalledWith(2, 'reload.mcp', { always: true, confirm: true }, MCP_RELOAD_TIMEOUT_MS)
    expect(request).toHaveBeenCalledTimes(2)
  })

  it('does not retry an unrelated failure', async () => {
    const request = vi.fn().mockRejectedValue(new Error('session busy — a turn is running'))
    const gateway = { request } as unknown as HermesGateway

    await expect(requestMcpReload(gateway, { profile: 'work' })).rejects.toThrow(/session busy/)
    expect(request).toHaveBeenCalledTimes(1)
  })

  it('stops after one retry when the fallback also rejects', async () => {
    const request = vi.fn().mockRejectedValue(new Error('invalid params for reload.mcp: profile: nope'))
    const gateway = { request } as unknown as HermesGateway

    await expect(requestMcpReload(gateway, { profile: 'work' })).rejects.toThrow(/invalid params/)
    expect(request).toHaveBeenCalledTimes(2)
  })

  it('refuses to call without a gateway', async () => {
    await expect(requestMcpReload(null)).rejects.toThrow(/not connected/)
  })
})

describe('mcpReloadSlashParams', () => {
  const cases: [string, Record<string, unknown>][] = [
    ['', { session_id: 's-1' }],
    ['now', { confirm: true, session_id: 's-1' }],
    ['approve', { confirm: true, session_id: 's-1' }],
    ['once', { confirm: true, session_id: 's-1' }],
    ['yes', { confirm: true, session_id: 's-1' }],
    ['always', { always: true, confirm: true, session_id: 's-1' }],
    ['bogus', { session_id: 's-1' }]
  ]

  it.each(cases)('maps %s to its reload parameters', (arg, expected) => {
    expect(mcpReloadSlashParams(arg, 's-1')).toEqual(expected)
  })
})

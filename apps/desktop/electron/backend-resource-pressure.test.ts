import assert from 'node:assert/strict'

import { test } from 'vitest'

import { waitForHermesReady } from './backend-health'
import { attachOrReserveSpawn } from './host-backend-attach'

const codes = ['ENOBUFS', 'EMFILE', 'ENFILE', 'EADDRNOTAVAIL']

test('readiness stops on resource exhaustion without spending the boot polling budget', async () => {
  for (const code of codes) {
    for (const alreadyBound of [false, true]) {
      for (const legacy of [false, true]) {
        const error = Object.assign(new Error(`connect ${code} 127.0.0.1:2802`), { code })
        let time = 0
        let probes = 0
        let sleeps = 0

        const fail = async () => {
          probes += 1
          throw error
        }

        await assert.rejects(
          waitForHermesReady('http://127.0.0.1:2802', {
            alreadyBound,
            fetchPublicJson: legacy
              ? async () => {
                  throw new Error('404: missing')
                }
              : fail,
            fetchJson: fail,
            now: () => time,
            sleep: async ms => {
              sleeps += 1
              time += ms
            }
          }),
          caught => caught === error
        )
        assert.equal(probes, 1, `${code}: one failed probe, including legacy status`)
        assert.equal(sleeps, 0, `${code}: no retry while the host cannot open sockets`)
      }
    }
  }
})

test('host attach cannot turn resource exhaustion into a reservation to spawn another backend', async () => {
  const ledger = JSON.stringify([{ pid: 4711, port: 2802, host: '127.0.0.1', purpose: 'serve' }])

  for (const code of codes) {
    for (const stage of ['token', 'readiness', 'identity']) {
      const transportError = Object.assign(new Error(`connect ${code}`), { code })
      const error = stage === 'readiness' ? transportError : new TypeError('fetch failed', { cause: transportError })
      let gateClaims = 0
      let time = 0

      const failAt = async <T>(rung: string, result: T): Promise<T> => {
        if (stage === rung) {
          throw error
        }

        return result
      }

      await assert.rejects(
        attachOrReserveSpawn(
          { isolated: false, ledgerPath: '/unused-ledger' },
          {
            readLedger: () => ledger,
            resolveServedToken: () => failAt('token', 'session-token'),
            waitForReady: () => failAt('readiness', undefined),
            expectedCodeIdentity: async () => 'current-commit',
            backendCodeIdentity: () => failAt('identity', 'current-commit'),
            probeWebSocket: () => failAt('websocket', { ok: true }),
            log: () => {}
          },
          {
            now: () => time,
            read: () => null,
            take: () => {
              gateClaims += 1

              return () => {}
            },
            sleep: async ms => {
              time += ms
            }
          }
        ),
        caught => caught === error
      )
      assert.equal(gateClaims, 0, `${code} at ${stage} does not prove the backend is gone`)
    }
  }
})

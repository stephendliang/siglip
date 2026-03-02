#!/usr/bin/env python3
"""Provision a Verda B200 instance, push+clone, build, run, fetch results.

Safety: refuses to provision if you already have running/provisioning instances.
Tracks instance ID in .remote_instance so you can't lose track of it.

Setup:
    pip install verda
    export VERDA_CLIENT_ID=...
    export VERDA_CLIENT_SECRET=...

Usage:
    # Provision + run default build+test
    python3 remote.py

    # Provision + run grid search
    python3 remote.py --sweep --tier all

    # Provision + run custom command
    python3 remote.py --cmd 'make && ./siglip_vision'

    # Re-use running instance (auto-detected from .remote_instance or API)
    python3 remote.py --cmd 'cd siglip && make && ./siglip_vision'

    # Destroy instance when done (default: leave running)
    python3 remote.py --destroy

    # Just provision and print SSH command (no build/run)
    python3 remote.py --provision-only

    # List running instances
    python3 remote.py --list

    # Destroy current tracked instance
    python3 remote.py --down

    # Destroy a specific instance by ID
    python3 remote.py --destroy-id abc-123-def

    # Force provision even if instances exist (you'd better know what you're doing)
    python3 remote.py --force-provision
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_URL = 'git@github.com:stephendliang/siglip.git'
REPO_DIR = 'siglip'
INSTANCE_TYPE = '1B200.30V'
IMAGE = 'ubuntu-24.04-cuda-12.8-open-docker'
HOSTNAME = 'siglip-bench'
SSH_USER = 'root'
SSH_KEY = os.path.expanduser('~/.ssh/id_ed25519')
BOOT_TIMEOUT = 300  # seconds to wait for SSH
POLL_INTERVAL = 10
RUN_TIMEOUT = 3600  # 1hr max for any remote command (prevents billing runaway)

# State file: tracks the instance we provisioned so we can't lose it
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.remote_instance')

ACTIVE_STATUSES = {'running', 'provisioning', 'ordered', 'restoring',
                   'starting_hibernation', 'hibernating'}


def get_client():
    try:
        from verda import VerdaClient
    except ImportError:
        print('Error: verda SDK not installed. Run: pip install verda', file=sys.stderr)
        sys.exit(1)

    client_id = os.environ.get('VERDA_CLIENT_ID')
    client_secret = os.environ.get('VERDA_CLIENT_SECRET')
    if not client_id or not client_secret:
        print('Error: set VERDA_CLIENT_ID and VERDA_CLIENT_SECRET env vars', file=sys.stderr)
        sys.exit(1)

    return VerdaClient(client_id, client_secret)


# ── State file management ──

def save_state(instance_id, ip=None):
    """Write instance ID to state file so we can find it later."""
    state = {'instance_id': instance_id, 'ip': ip, 'created': time.time()}
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def load_state():
    """Load tracked instance from state file. Returns (instance_id, ip) or (None, None)."""
    if not os.path.exists(STATE_FILE):
        return None, None
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        return state.get('instance_id'), state.get('ip')
    except (json.JSONDecodeError, KeyError):
        return None, None


def clear_state():
    """Remove state file."""
    if os.path.exists(STATE_FILE):
        os.unlink(STATE_FILE)


def get_active_instances(verda):
    """Return list of instances in non-terminal states."""
    all_inst = verda.instances.get()
    return [i for i in all_inst if i.status in ACTIVE_STATUSES]


def find_running_instance(verda):
    """Try to find a usable running instance. Checks state file first, then API."""
    # 1. Check state file
    tracked_id, tracked_ip = load_state()
    if tracked_id:
        try:
            inst = verda.instances.get_by_id(tracked_id)
            if inst.status == 'running' and inst.ip:
                return inst
            if inst.status in ACTIVE_STATUSES:
                # Still booting or hibernating — return it but caller must handle
                return inst
            # Dead instance in state file — clean up
            clear_state()
        except Exception:
            # Instance gone — clean up
            clear_state()

    # 2. Check API for any running instances with our hostname
    active = get_active_instances(verda)
    for inst in active:
        if inst.hostname == HOSTNAME and inst.status == 'running' and inst.ip:
            save_state(inst.id, inst.ip)
            return inst

    return None


# ── Core operations ──

def list_instances():
    verda = get_client()
    instances = verda.instances.get()
    if not instances:
        print('No instances.')
        return
    tracked_id, _ = load_state()
    for inst in instances:
        marker = ' ← tracked' if inst.id == tracked_id else ''
        print(f'  {inst.id}  {inst.instance_type:20s}  {inst.status:15s}  '
              f'{inst.ip or "no-ip":>16s}  {inst.hostname}{marker}')


def destroy_instance(instance_id):
    verda = get_client()
    print(f'Destroying instance {instance_id}...')
    verda.instances.action(instance_id, verda.actions.DELETE)
    # Clear state if this was our tracked instance
    tracked_id, _ = load_state()
    if tracked_id == instance_id:
        clear_state()
    print('Done.')


def destroy_tracked():
    """Destroy the instance in .remote_instance."""
    tracked_id, _ = load_state()
    if not tracked_id:
        print('No tracked instance. Use --list to see instances, --destroy-id ID to destroy one.')
        return
    destroy_instance(tracked_id)


def provision(gpu_type, force=False):
    """Create a B200 instance and wait for it to be running with an IP."""
    verda = get_client()

    # Safety: check for existing instances
    active = get_active_instances(verda)
    if active and not force:
        print('ABORT: you already have active instances:', file=sys.stderr)
        for inst in active:
            cost = ''
            try:
                cost = f'  ${inst.price_per_hour:.2f}/hr'
            except (AttributeError, TypeError):
                pass
            print(f'  {inst.id}  {inst.instance_type:20s}  {inst.status:15s}  '
                  f'{inst.ip or "no-ip":>16s}  {inst.hostname}{cost}', file=sys.stderr)
        print(f'\nTo reuse a running instance, just run without --force-provision.',
              file=sys.stderr)
        print(f'To provision anyway: --force-provision', file=sys.stderr)
        print(f'To destroy first:    --down  or  --destroy-id ID', file=sys.stderr)
        sys.exit(1)

    # Check state file for stale tracked instance
    tracked_id, _ = load_state()
    if tracked_id and not force:
        try:
            inst = verda.instances.get_by_id(tracked_id)
            if inst.status in ACTIVE_STATUSES:
                print(f'ABORT: tracked instance {tracked_id} still {inst.status}.',
                      file=sys.stderr)
                print(f'Use --down to destroy it first, or --force-provision.', file=sys.stderr)
                sys.exit(1)
        except Exception:
            clear_state()  # stale reference

    # Get SSH keys
    ssh_keys = verda.ssh_keys.get()
    if not ssh_keys:
        print('Error: no SSH keys configured in Verda. Add one at console.verda.com',
              file=sys.stderr)
        sys.exit(1)
    ssh_key_ids = [k.id for k in ssh_keys]
    print(f'Using SSH keys: {[k.id[:8] + "..." for k in ssh_keys]}')

    # Check availability
    available = verda.instances.is_available(gpu_type)
    if not available:
        print(f'Error: {gpu_type} not available right now', file=sys.stderr)
        avail = verda.instances.get_availabilities()
        b200_avail = [a for a in avail if 'B200' in str(a.get('instance_type', ''))]
        if b200_avail:
            print(f'Available B200 types: {b200_avail}', file=sys.stderr)
        sys.exit(1)

    print(f'Provisioning {gpu_type} ({IMAGE})...')
    instance = verda.instances.create(
        instance_type=gpu_type,
        image=IMAGE,
        ssh_key_ids=ssh_key_ids,
        hostname=HOSTNAME,
        description='siglip megakernel bench',
    )
    print(f'Instance created: {instance.id} (status: {instance.status})')

    # Track immediately so we can't lose it
    save_state(instance.id)

    # Wait for running + IP
    t0 = time.time()
    while True:
        instance = verda.instances.get_by_id(instance.id)
        elapsed = time.time() - t0
        if instance.status == 'running' and instance.ip:
            print(f'\nInstance running: {instance.ip} ({elapsed:.0f}s)')
            save_state(instance.id, instance.ip)
            break
        if instance.status == 'error':
            print(f'\nError: instance entered error state', file=sys.stderr)
            print(f'Instance ID (may need manual cleanup): {instance.id}', file=sys.stderr)
            sys.exit(1)
        if elapsed > BOOT_TIMEOUT:
            print(f'\nError: timeout waiting for instance ({BOOT_TIMEOUT}s)', file=sys.stderr)
            print(f'Instance {instance.id} still in state: {instance.status}', file=sys.stderr)
            print(f'NOT destroying — check manually: python3 remote.py --list', file=sys.stderr)
            sys.exit(1)
        print(f'  {instance.status}... ({elapsed:.0f}s)', end='\r', flush=True)
        time.sleep(POLL_INTERVAL)

    return instance


def wait_for_ssh(ip):
    """Wait until SSH is accepting connections."""
    print('Waiting for SSH...', end=' ', flush=True)
    t0 = time.time()
    while time.time() - t0 < BOOT_TIMEOUT:
        try:
            ret = subprocess.run(
                ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
                 '-o', 'ConnectTimeout=5', '-o', 'BatchMode=yes',
                 f'{SSH_USER}@{ip}', 'true'],
                capture_output=True, timeout=15
            )
            if ret.returncode == 0:
                print(f'OK ({time.time() - t0:.0f}s)')
                return True
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
    print('TIMEOUT')
    return False


def ssh_cmd(ip):
    """Return base SSH command list."""
    return ['ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
            '-o', 'BatchMode=yes', '-A', f'{SSH_USER}@{ip}']


def run_remote(ip, cmd, stream=True, timeout=RUN_TIMEOUT):
    """Run a command on the remote host. If stream=True, print output live."""
    full_cmd = ssh_cmd(ip) + [cmd]
    if stream:
        ret = subprocess.run(full_cmd, timeout=timeout)
        return ret.returncode, '', ''
    else:
        ret = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return ret.returncode, ret.stdout, ret.stderr


def git_push_local():
    """Ensure local changes are pushed to GitHub."""
    ret = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    dirty = [l for l in ret.stdout.strip().splitlines()
             if not l.startswith('??')]
    if dirty:
        print('Warning: uncommitted tracked changes:')
        for l in dirty:
            print(f'  {l}')
        print('These will NOT be on the remote. Commit first or they will be missing.')
        resp = input('Continue anyway? [y/N] ')
        if resp.lower() != 'y':
            sys.exit(0)

    branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    print(f'Pushing {branch} to origin...')
    ret = subprocess.run(['git', 'push', 'origin', branch], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f'git push failed:\n{ret.stderr}', file=sys.stderr)
        sys.exit(1)
    print('Pushed.')


def setup_remote(ip):
    """Clone repo (with submodules) on the remote host."""
    rc, out, _ = run_remote(ip, f'test -d {REPO_DIR} && echo exists', stream=False)
    if 'exists' in out:
        print('Repo already exists on remote, pulling latest...')
        run_remote(ip, f'cd {REPO_DIR} && git pull && git submodule update --init --recursive',
                   stream=True)
    else:
        print('Cloning repo on remote...')
        run_remote(ip, f'git clone --recursive {REPO_URL}', stream=True)


def fetch_file(ip, remote_path, local_path):
    """SCP a file from the remote."""
    cmd = ['scp', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
           f'{SSH_USER}@{ip}:{remote_path}', local_path]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f'Warning: SCP failed for {remote_path}: {ret.stderr.strip()}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Remote B200 test runner for siglip megakernel')

    # Instance lifecycle
    parser.add_argument('--list', action='store_true', help='List all instances')
    parser.add_argument('--destroy-id', metavar='ID', help='Destroy a specific instance')
    parser.add_argument('--down', action='store_true',
                        help='Destroy the tracked instance (.remote_instance)')
    parser.add_argument('--provision-only', action='store_true',
                        help='Just provision and print SSH command')
    parser.add_argument('--instance-id', metavar='ID',
                        help='Use a specific instance instead of auto-detect')
    parser.add_argument('--destroy', action='store_true',
                        help='Destroy instance after run completes')
    parser.add_argument('--force-provision', action='store_true',
                        help='Provision new instance even if others exist')

    # What to run
    parser.add_argument('--cmd', metavar='CMD',
                        help='Custom command to run on remote (in repo dir)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run grid_search.py')
    parser.add_argument('--tier', default='all',
                        help='Tier for --sweep (1, 2, 3, all, or full-cross)')
    parser.add_argument('--sweep-args', default='',
                        help='Extra args for grid_search.py')

    # Options
    parser.add_argument('--no-push', action='store_true',
                        help='Skip git push (assume remote is up to date)')
    parser.add_argument('--fetch-csv', action='store_true',
                        help='Fetch sweep_results.csv after sweep')
    parser.add_argument('--gpu', default=INSTANCE_TYPE,
                        help=f'Instance type (default: {INSTANCE_TYPE})')

    args = parser.parse_args()

    # ── Simple commands ──
    if args.list:
        list_instances()
        return

    if args.destroy_id:
        destroy_instance(args.destroy_id)
        return

    if args.down:
        destroy_tracked()
        return

    # ── Resolve instance: reuse existing or provision ──
    verda = get_client()
    instance = None
    ip = None

    if args.instance_id:
        # Explicit instance ID
        instance = verda.instances.get_by_id(args.instance_id)
        ip = instance.ip
        if not ip:
            print(f'Error: instance {args.instance_id} has no IP (status: {instance.status})',
                  file=sys.stderr)
            sys.exit(1)
        save_state(instance.id, ip)
        print(f'Using specified instance: {ip}')

    elif args.force_provision:
        # Forced provision
        instance = provision(args.gpu, force=True)
        ip = instance.ip

    else:
        # Auto-detect: try to reuse, provision only if nothing exists
        existing = find_running_instance(verda)
        if existing and existing.status == 'running' and existing.ip:
            instance = existing
            ip = existing.ip
            print(f'Reusing running instance: {ip} ({instance.hostname}, {instance.instance_type})')
            save_state(instance.id, ip)
        elif existing:
            # Instance exists but not ready
            print(f'Found instance {existing.id} in state: {existing.status}')
            print(f'Wait for it to finish or destroy it: --down')
            sys.exit(1)
        else:
            # Nothing running — provision
            print('No running instances found, provisioning...')
            instance = provision(args.gpu)
            ip = instance.ip

    instance_id = instance.id
    ssh_str = f'ssh -i {SSH_KEY} {SSH_USER}@{ip}'
    print(f'\nSSH: {ssh_str}')

    if args.provision_only:
        print(f'Instance ID: {instance_id}')
        print(f'To destroy: python3 remote.py --down')
        return

    # Everything below is wrapped in try/finally so --destroy fires even on
    # failure, ctrl-C, or timeout — prevents billing runaway.
    rc = 1
    try:
        # Wait for SSH
        if not wait_for_ssh(ip):
            print('Could not connect via SSH. Instance may still be booting.')
            print(f'Instance ID: {instance_id}')
            return

        # Push and setup
        if not args.no_push:
            git_push_local()
        setup_remote(ip)

        # Determine command
        if args.cmd:
            remote_cmd = f'cd {REPO_DIR} && {args.cmd}'
        elif args.sweep:
            tier_arg = '--full-cross' if args.tier == 'full-cross' else f'--tier {args.tier}'
            remote_cmd = f'cd {REPO_DIR} && python3 grid_search.py {tier_arg} {args.sweep_args}'
        else:
            remote_cmd = f'cd {REPO_DIR} && make && ./siglip_vision'

        print(f'\nRunning: {remote_cmd}\n')
        print('=' * 60)
        rc, _, _ = run_remote(ip, remote_cmd, stream=True)
        print('=' * 60)
        print(f'\nExit code: {rc}')

        # Fetch results
        if args.fetch_csv and args.sweep:
            local_csv = 'sweep_results_remote.csv'
            print(f'Fetching sweep_results.csv → {local_csv}')
            fetch_file(ip, f'{REPO_DIR}/sweep_results.csv', local_csv)

    except subprocess.TimeoutExpired:
        print(f'\nTIMEOUT: remote command exceeded {RUN_TIMEOUT}s', file=sys.stderr)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    finally:
        if args.destroy:
            print(f'\nDestroying instance (--destroy)...')
            destroy_instance(instance_id)
        else:
            try:
                cost = f'${instance.price_per_hour:.2f}/hr'
            except (AttributeError, TypeError):
                cost = 'unknown rate'
            print(f'\nInstance still running: {ip} ({cost})')
            print(f'  SSH:     {ssh_str}')
            print(f'  Destroy: python3 remote.py --down')


if __name__ == '__main__':
    main()

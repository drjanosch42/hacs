// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

#pragma once
#ifndef VALIDATION_SHARD_H
#define VALIDATION_SHARD_H

// (S24 SHARD, VALSTREAM item 5; docs/implementation/slot24/VALSTREAM_audit_1030.md Sec. 3)
// The partition behind `--all --shard i/N` / `--all-full --shard i/N`.
//
// The roster of a tier (`--all`: fast + standard; `--all-full`: every tier) is cut into N shards, run as N
// processes.  The partition is a pure function of (roster in registry order, N):
//   * units: every name is its own unit, except the CO-LOCATION GROUPS below, each one unit;
//     the device group (every name that creates the CUDA context) is one of them, so all device work
//     lands in ONE shard;
//   * weight of a unit = the sum of its names' expected wall times (the committed table
//     ValidationShardWeights.inc, milliseconds; a name without a row gets kShardDefaultWeightMs);
//   * LPT greedy: units by weight descending (ties: earlier registry position first), each to the
//     currently lightest shard (ties: the lowest shard index);
//   * within a shard the names run in registry order.
// So the union over i = 1..N is exactly the roster and every name is in exactly one shard.  Integer
// weights only: the partition is the same on every machine and every build of the same roster.
//
// Shard indices are 1-BASED: `--shard 1/4` .. `--shard 4/4`.  N is in [1, kShardMaxCount].

#include <cstdint>
#include <string>
#include <vector>

constexpr uint32_t kShardMaxCount = 64;
constexpr uint32_t kShardDefaultWeightMs = 5000;  // a name not in the weights table (new since the last measurement)

struct ShardSpec {
    uint32_t index = 1;  // 1-based
    uint32_t count = 1;
};

// Parses "i/N" (decimal, 1 <= i <= N <= kShardMaxCount, no sign, no spaces).  false + `err` otherwise.
bool ParseShardSpec(const std::string& text, ShardSpec& out, std::string& err);

// Expected wall time of one name in milliseconds (the table, else kShardDefaultWeightMs).
uint32_t ShardWeightMs(const std::string& name);
// Whether `name` has a row in the weights table.
bool ShardHasWeight(const std::string& name);

// True iff `name` is in the device group (pinned to the device shard).
bool ShardIsDevicePinned(const std::string& name);

// The co-location groups (the device group first).  A group's names that are not in the roster are ignored.
const std::vector<std::vector<std::string>>& ShardColocationGroups();

struct ShardPartition {
    uint32_t count = 1;
    std::vector<uint32_t> shard_of;       // per roster position: its shard, 1-based
    std::vector<uint64_t> load_ms;        // per shard (index 0 = shard 1): the summed weights
    uint32_t device_shard = 0;            // 1-based; 0 when the roster has no device-pinned name
};

// The partition of `roster` (registry order) into `count` shards, as described above.
ShardPartition PartitionShards(const std::vector<std::string>& roster, uint32_t count);

// The partition self-check run by ValidationHarnessCatchValidation: for N in {1, 2, 3, 4, 5, 8, 16} and
// both rosters of the live registry (--all, --all-full), the union is the roster, the shards are disjoint,
// every device-pinned name sits in one shard, every co-location group in one shard, the order within a
// shard is the registry order, the partition is reproducible, and malformed specs are refused.  Prints
// one `SHARD_PARTITION ...` line per roster; returns the number of failing checks.
uint32_t ShardPartitionSelfCheck(uint32_t& checks);

#endif // VALIDATION_SHARD_H

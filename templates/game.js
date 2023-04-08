// 游戏配置
const PLAYER_HP = 5; // 玩家初始生命值
const ENEMY_HP = 5; // 电脑初始生命值
const PLAYER_ATTACK = 1; // 玩家初始攻击力
const ENEMY_ATTACK = 1; // 电脑初始攻击力

// 游戏状态
let playerHp = PLAYER_HP; // 玩家当前生命值
let enemyHp = ENEMY_HP; // 电脑当前生命值
let playerAttack = PLAYER_ATTACK; // 玩家当前攻击力
let enemyAttack = ENEMY_ATTACK; // 电脑当前攻击力

let playerDefenseSuccessRate = 1; // 玩家防御成功率
let enemyDefenseSuccessRate = 1; // 电脑防御成功率

let isPlayerDefending = false; // 玩家是否正在防御
let isEnemyDefending = false; // 电脑是否正在防御

let playerHistory = []; // 玩家历史动作
let enemyHistory = []; // 电脑历史动作

// 攻击函数
function attack(isPlayer) {
  if (isPlayer) { // 玩家攻击
    if (!isEnemyDefending || Math.random() < enemyDefenseSuccessRate) { // 电脑未防御或防御失败
      enemyHp -= playerAttack; // 电脑扣除生命值
      enemyHistory.push("受到攻击 " + playerAttack + " 点伤害"); // 记录历史动作
    } else { // 电脑防御成功
      playerHp += enemyAttack; // 玩家回复生命值
      playerHistory.push("攻击被防御，回复 " + enemyAttack + " 点生命值"); // 记录历史动作
    }
    playerAttack = PLAYER_ATTACK; // 玩家攻击力重置为初始值
    isPlayerDefending = false; // 玩家不再处于防御状态
  } else { // 电脑攻击
    if (!isPlayerDefending || Math.random() < playerDefenseSuccessRate) { // 玩家未防御或防御失败
      playerHp -= enemyAttack; // 玩家扣除生命值
      playerHistory.push("受到攻击 " + enemyAttack + " 点伤害"); // 记录历史动作
    } else { // 玩家防御成功
      enemyHp += playerAttack; // 电脑回复生命值
      enemyHistory.push("攻击被防御，回复 " + playerAttack + " 点生命值"); // 记录历史动作
    }
    enemyAttack = ENEMY_ATTACK; // 电脑攻击力重置为初始值
    isEnemyDefending = false; // 电脑不再处于防御状态
  }

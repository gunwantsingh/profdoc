// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title Advanced Federated Learning Contract
 * @dev Manages federated learning rounds with on-chain coordination and off-chain storage
 * @dev Uses a hybrid architecture for scalable machine learning on blockchain
 */
contract AdvancedFederatedLearning is Ownable {
    // ===== STRUCTS =====
    struct Participant {
        address nodeAddress;
        bool isActive;
        uint256 reputation;
        uint256 lastParticipation;
    }

    struct TrainingRound {
        uint256 roundId;
        bytes32 globalModelHash;
        string globalModelUri;
        uint256 startBlock;
        uint256 endBlock;
        RoundStatus status;
        uint256 participantCount;
        mapping(address => bool) hasParticipated;
        mapping(address => bytes32) participantHashes;
    }

    // ===== ENUMS =====
    enum RoundStatus { NOT_STARTED, ACTIVE, AGGREGATING, COMPLETED }

    // ===== STATE VARIABLES =====
    uint256 public currentRoundId;
    uint256 public minimumParticipants;
    uint256 public roundDurationBlocks;
    
    mapping(uint256 => TrainingRound) public rounds;
    mapping(address => Participant) public participants;

    // ===== EVENTS =====
    event RoundStarted(uint256 indexed roundId, bytes32 initialModelHash, string modelUri);
    event ParticipantRegistered(address indexed participant);
    event ModelSubmitted(uint256 indexed roundId, address indexed participant, bytes32 modelHash);
    event RoundCompleted(uint256 indexed roundId, bytes32 aggregatedHash, string modelUri);
    event ParticipantSlashed(address indexed participant, uint256 amount);
    event RewardDistributed(address indexed participant, uint256 amount);

    // ===== MODIFIERS =====
    modifier onlyActiveParticipant() {
        require(participants[msg.sender].isActive, "Not an active participant");
        _;
    }

    modifier validRound(uint256 roundId) {
        require(roundId <= currentRoundId, "Invalid round ID");
        _;
    }

    // ===== CONSTRUCTOR =====
    constructor(
        uint256 _minimumParticipants,
        uint256 _roundDurationBlocks
    ) {
        minimumParticipants = _minimumParticipants;
        roundDurationBlocks = _roundDurationBlocks;
        currentRoundId = 0;
    }

    // ===== EXTERNAL FUNCTIONS =====

    /**
     * @dev Register as a participant in the federated learning network
     * @param initialReputation Starting reputation score for the participant
     */
    function registerParticipant(uint256 initialReputation) external {
        require(!participants[msg.sender].isActive, "Already registered");
        
        participants[msg.sender] = Participant({
            nodeAddress: msg.sender,
            isActive: true,
            reputation: initialReputation,
            lastParticipation: 0
        });

        emit ParticipantRegistered(msg.sender);
    }

    /**
     * @dev Start a new training round with initial model parameters
     * @param initialModelHash Hash of the initial global model weights
     * @param modelUri Off-chain storage URI for the model weights
     */
    function startNewRound(
        bytes32 initialModelHash,
        string calldata modelUri
    ) external onlyOwner {
        currentRoundId++;
        
        TrainingRound storage newRound = rounds[currentRoundId];
        newRound.roundId = currentRoundId;
        newRound.globalModelHash = initialModelHash;
        newRound.globalModelUri = modelUri;
        newRound.startBlock = block.number;
        newRound.endBlock = block.number + roundDurationBlocks;
        newRound.status = RoundStatus.ACTIVE;

        emit RoundStarted(currentRoundId, initialModelHash, modelUri);
    }

    /**
     * @dev Submit local model updates for the current round
     * @param modelHash Hash of the local model weights
     */
    function submitModel(bytes32 modelHash) external onlyActiveParticipant {
        TrainingRound storage currentRound = rounds[currentRoundId];
        require(currentRound.status == RoundStatus.ACTIVE, "Round not active");
        require(block.number <= currentRound.endBlock, "Round ended");
        require(!currentRound.hasParticipated[msg.sender], "Already participated");

        currentRound.hasParticipated[msg.sender] = true;
        currentRound.participantHashes[msg.sender] = modelHash;
        currentRound.participantCount++;

        participants[msg.sender].lastParticipation = block.number;

        emit ModelSubmitted(currentRoundId, msg.sender, modelHash);
    }

    /**
     * @dev Complete the round and set the new aggregated model
     * @param aggregatedModelHash Hash of the aggregated model weights
     * @param modelUri Off-chain storage URI for the aggregated model
     */
    function completeRound(
        bytes32 aggregatedModelHash,
        string calldata modelUri
    ) external onlyOwner {
        TrainingRound storage currentRound = rounds[currentRoundId];
        require(currentRound.status == RoundStatus.ACTIVE, "Round not active");
        require(block.number > currentRound.endBlock, "Round not ended");
        require(currentRound.participantCount >= minimumParticipants, "Insufficient participation");

        currentRound.globalModelHash = aggregatedModelHash;
        currentRound.globalModelUri = modelUri;
        currentRound.status = RoundStatus.COMPLETED;

        emit RoundCompleted(currentRoundId, aggregatedModelHash, modelUri);
    }

    // ===== VIEW FUNCTIONS =====

    /**
     * @dev Check if a participant has submitted for a specific round
     */
    function hasParticipated(uint256 roundId, address participant) 
        external 
        view 
        validRound(roundId) 
        returns (bool) 
    {
        return rounds[roundId].hasParticipated[participant];
    }

    /**
     * @dev Get participant's submitted model hash for a round
     */
    function getParticipantHash(uint256 roundId, address participant)
        external
        view
        validRound(roundId)
        returns (bytes32)
    {
        return rounds[roundId].participantHashes[participant];
    }

    /**
     * @dev Get current round status and information
     */
    function getRoundInfo(uint256 roundId)
        external
        view
        validRound(roundId)
        returns (
            uint256 roundId_,
            bytes32 globalModelHash,
            string memory globalModelUri,
            uint256 participantCount,
            RoundStatus status
        )
    {
        TrainingRound storage round = rounds[roundId];
        return (
            round.roundId,
            round.globalModelHash,
            round.globalModelUri,
            round.participantCount,
            round.status
        );
    }

    // ===== ADMIN FUNCTIONS =====

    /**
     * @dev Update round configuration parameters
     */
    function updateRoundConfig(
        uint256 newMinimumParticipants,
        uint256 newRoundDuration
    ) external onlyOwner {
        minimumParticipants = newMinimumParticipants;
        roundDurationBlocks = newRoundDuration;
    }

    /**
     * @dev Slash a participant's reputation for malicious behavior
     */
    function slashParticipant(address participant, uint256 amount) external onlyOwner {
        require(participants[participant].isActive, "Invalid participant");
        participants[participant].reputation -= amount;
        
        if (participants[participant].reputation == 0) {
            participants[participant].isActive = false;
        }

        emit ParticipantSlashed(participant, amount);
    }

    /**
     * @dev Reward a participant for good behavior
     */
    function rewardParticipant(address participant, uint256 amount) external onlyOwner {
        require(participants[participant].isActive, "Invalid participant");
        participants[participant].reputation += amount;
        emit RewardDistributed(participant, amount);
    }
}

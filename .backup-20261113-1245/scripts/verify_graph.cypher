// 대표 관계 타입과 개수
MATCH ()-[r]->()
RETURN type(r) AS relation_type, count(r) AS count
ORDER BY count DESC;

// 특정 작물에서 시작하는 2-hop 경로
MATCH path = (crop:Crop)-[*2]-(target)
RETURN path
LIMIT 20;

// 방향을 고려한 2-hop 예시
MATCH path = (crop:Crop)-[:HAS_VARIETY]->(variety:Variety)
             -[:SUITABLE_FOR_REGION]->(region:Region)
RETURN crop.name, variety.name, region.name, path
LIMIT 20;

// 2-hop 이상 경로(최대 4-hop)
MATCH path = (start:Entity)-[*2..4]-(end:Entity)
RETURN path
LIMIT 20;

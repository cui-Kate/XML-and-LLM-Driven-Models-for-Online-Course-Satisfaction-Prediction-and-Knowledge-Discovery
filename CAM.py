
import json
import requests
import time
import numpy as np
import re
import os
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import chardet
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SiliconFlow API configuration
SILICONFLOW_API_KEY = "your key"
EMBEDDING_URL = "your api"
EMBEDDING_MODEL = "your model"

def get_siliconflow_embedding(text, max_retries=3):
    """Get text embeddings from SiliconFlow API"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(EMBEDDING_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return np.array(data['data'][0]['embedding'])
        except (requests.exceptions.RequestException, KeyError) as e:
            wait_time = 2 ** attempt
            logger.warning(f"API request failed, retrying in {wait_time}s... Error: {str(e)}")
            time.sleep(wait_time)
        except json.JSONDecodeError as e:
            logger.error(f"API response JSON decode failed: {str(e)}")
            return None
    
    logger.error("API request failed after max retries")
    return None

def robust_json_parse(json_str, segment_id):
    """Robust JSON parsing with error handling"""
    # Remove control characters
    json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Standard parsing failed, fixing JSON for segment {segment_id}...")
        try:
            # Fix common errors
            fixed_str = re.sub(r'([\{\,])\s*([a-zA-Z_][\w]*)\s*:', r'\1"\2":', json_str)
            fixed_str = re.sub(r',\s*([\]\}])', r'\1', fixed_str)
            fixed_str = fixed_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            return json.loads(fixed_str)
        except json.JSONDecodeError as e2:
            logger.error(f"Fix failed for segment {segment_id}: {str(e2)}")
            return None

def parse_cam_data(file_content):
    """Parse CAM data from text file content"""
    segments = {}
    current_segment = None
    current_content = []
    
    # Process file line by line
    for line in file_content.split('\n'):
        stripped_line = line.strip()
        
        # Detect segment start (001, 002, etc.)
        if re.match(r'^\d{3}$', stripped_line):
            if current_segment is not None:
                segments[current_segment] = '\n'.join(current_content)
                current_content = []
            current_segment = stripped_line
        elif current_segment is not None:
            if stripped_line and not stripped_line.startswith('#'):
                current_content.append(line)
    
    # Add last segment
    if current_segment is not None and current_content:
        segments[current_segment] = '\n'.join(current_content)
    
    # Parse JSON data for each segment
    cam_data = []
    for segment_id, segment_content in segments.items():
        segment_content = segment_content.strip()
        if not segment_content:
            logger.warning(f"Segment {segment_id} is empty")
            continue
        
        segment_data = robust_json_parse(segment_content, segment_id)
        if segment_data is None:
            logger.error(f"Skipping segment {segment_id}, JSON parse failed")
            continue
        
        # Process different data structures
        if isinstance(segment_data, list):
            for item in segment_data:
                if isinstance(item, dict) and "amc" in item and isinstance(item["amc"], dict):
                    item["amc"]["segment_id"] = segment_id
                    cam_data.append(item["amc"])
                else:
                    logger.warning(f"Item in segment {segment_id} missing valid amc field")
        elif isinstance(segment_data, dict) and "amc" in segment_data:
            segment_data["amc"]["segment_id"] = segment_id
            cam_data.append(segment_data["amc"])
        else:
            logger.warning(f"Segment {segment_id} has unexpected structure")
    
    logger.info(f"Successfully loaded {len(cam_data)} CAM entries")
    return cam_data

def detect_file_encoding(file_path):
    """Detect file encoding"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']

def load_cam_data(file_path):
    """Load and parse CAM data file"""
    try:
        encoding = detect_file_encoding(file_path)
        logger.info(f"Detected file encoding: {encoding}")
        
        with open(file_path, "r", encoding=encoding, errors='replace') as f:
            file_content = f.read()
            return parse_cam_data(file_content)
    except IOError as e:
        logger.error(f"File load error: {str(e)}")
        return []
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error: {str(e)}")
        return []

def semantic_generalization(mechanisms, top_k=3):
    """Semantic mechanism generalization"""
    if not mechanisms:
        return "Composite mechanism"
    
    counter = Counter(mechanisms)
    most_common = counter.most_common(1)[0][0]
    
    parts = re.split(r'[→\+\-\>\<\=]', most_common)
    if parts:
        return " → ".join([p.strip() for p in parts[:2]])
    
    return most_common

def cluster_mechanisms(cams, similarity_threshold=0.70, top_k=8):
    """Cluster mechanisms using semantic embeddings"""
    if not cams or len(cams) < 10:
        logger.warning("Insufficient data, skipping clustering")
        return []
    
    mechanisms = [cam["mechanism"] for cam in cams]
    logger.info(f"Generating embeddings for {len(mechanisms)} mechanisms...")
    
    embeddings = []
    for i, mech in enumerate(mechanisms):
        if i % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(mechanisms)}")
        
        embedding = get_siliconflow_embedding(mech)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append(np.zeros(768))
            logger.warning(f"Embedding failed for mechanism: {mech[:30]}...")
    
    logger.info("Calculating similarity matrix...")
    sim_matrix = cosine_similarity(embeddings)
    
    clusters = defaultdict(list)
    visited = set()
    
    for i in range(len(mechanisms)):
        if i in visited:
            continue
            
        cluster_id = f"cluster_{len(clusters)+1}"
        clusters[cluster_id].append(i)
        visited.add(i)
        
        for j in range(i+1, len(mechanisms)):
            if j not in visited and sim_matrix[i][j] > similarity_threshold:
                clusters[cluster_id].append(j)
                visited.add(j)
    
    logger.info(f"Identified {len(clusters)} mechanism clusters")
    
    filtered_clusters = {cid: idxs for cid, idxs in clusters.items() if len(idxs) > 1}
    logger.info(f"Filtered to {len(filtered_clusters)} valid clusters (size > 1)")
    
    results = []
    for cluster_id, indices in filtered_clusters.items():
        cluster_cams = [cams[i] for i in indices]
        cluster_mechanisms = [cams[i]["mechanism"] for i in indices]
        
        generalized_mech = semantic_generalization(cluster_mechanisms)
        
        contexts = Counter(cam["context"] for cam in cluster_cams)
        antecedents = Counter(cam["antecedent"] for cam in cluster_cams)
        segment_distribution = Counter(cam.get("segment_id", "unknown") for cam in cluster_cams)
        
        mechanism_counter = Counter(cluster_mechanisms)
        representative_mech = mechanism_counter.most_common(1)[0][0]
        top_mechanisms = [mech for mech, _ in mechanism_counter.most_common(top_k)]
        
        results.append({
            "cluster_id": cluster_id,
            "generalized_mechanism": generalized_mech,
            "representative_mechanism": representative_mech,
            "mechanism_count": len(indices),
            "segment_distribution": segment_distribution.most_common(),
            "top_contexts": contexts.most_common(3),
            "top_antecedents": antecedents.most_common(3),
            "top_mechanisms": top_mechanisms
        })
    
    return sorted(results, key=lambda x: x["mechanism_count"], reverse=True)

def save_json_results(results, output_path):
    """Save clustering results as JSON"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_path}")
    except IOError as e:
        logger.error(f"Failed to save results: {str(e)}")

def save_text_report(results, output_path, top_n=10):
    """Save cluster analysis report as text file"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Cross-Context Mechanism Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            total_clusters = len(results)
            total_mechanisms = sum(cluster["mechanism_count"] for cluster in results)
            covered_segments = set()
            
            for cluster in results:
                covered_segments.update(seg for seg, _ in cluster["segment_distribution"])
            
            f.write(f"Overall Statistics:\n")
            f.write(f"- Identified Clusters: {total_clusters}\n")
            f.write(f"- Total Mechanisms: {total_mechanisms}\n")
            f.write(f"- Segments Covered: {len(covered_segments)}/10\n\n")
            
            # Top clusters
            f.write(f"Top {top_n} Clusters:\n\n")
            for i, cluster in enumerate(results[:top_n]):
                f.write(f"Cluster #{i+1} ({cluster['cluster_id']}):\n")
                f.write(f"- Generalized Mechanism: {cluster['generalized_mechanism']}\n")
                f.write(f"- Representative Mechanism: {cluster['representative_mechanism']}\n")
                f.write(f"- Cluster Size: {cluster['mechanism_count']} mechanisms\n")
                
                # Segment distribution
                f.write("- Segment Distribution:\n")
                for segment, count in cluster["segment_distribution"]:
                    f.write(f"  Segment {segment}: {count} mechanisms\n")
                
                # Contexts and antecedents
                f.write("- Typical Contexts:\n")
                for ctx, count in cluster["top_contexts"]:
                    f.write(f"  - {ctx} ({count} occurrences)\n")
                
                f.write("- Frequent Antecedents:\n")
                for ant, count in cluster["top_antecedents"]:
                    f.write(f"  - {ant} ({count} occurrences)\n")
                
                f.write("- Top Mechanism Examples:\n")
                for j, mech in enumerate(cluster["top_mechanisms"][:3]):
                    f.write(f"  {j+1}. {mech}\n")
                
                f.write("-" * 60 + "\n\n")
        
        logger.info(f"Report saved to: {output_path}")
    except IOError as e:
        logger.error(f"Failed to save report: {str(e)}")

def print_analysis_report(results, top_n=10):
    """Print analysis report to console"""
    if not results:
        logger.warning("No clustering results available")
        return
    
    print("\n" + "=" * 80)
    print("Cross-Context Mechanism Analysis Report")
    print("=" * 80)
    
    # Overall statistics
    total_clusters = len(results)
    total_mechanisms = sum(cluster["mechanism_count"] for cluster in results)
    covered_segments = set()
    
    for cluster in results:
        covered_segments.update(seg for seg, _ in cluster["segment_distribution"])
    
    print(f"\nOverall Statistics:")
    print(f"- Identified Clusters: {total_clusters}")
    print(f"- Total Mechanisms: {total_mechanisms}")
    print(f"- Segments Covered: {len(covered_segments)}/10")
    
    # Top clusters
    print(f"\nTop {top_n} Clusters:")
    for i, cluster in enumerate(results[:top_n]):
        print(f"\nCluster #{i+1} ({cluster['cluster_id']}):")
        print(f"- Generalized Mechanism: {cluster['generalized_mechanism']}")
        print(f"- Representative Mechanism: {cluster['representative_mechanism']}")
        print(f"- Cluster Size: {cluster['mechanism_count']} mechanisms")
        
        # Segment distribution
        print("- Segment Distribution:")
        for segment, count in cluster["segment_distribution"]:
            print(f"  Segment {segment}: {count} mechanisms")
        
        # Contexts and antecedents
        print("- Typical Contexts:")
        for ctx, count in cluster["top_contexts"]:
            print(f"  - {ctx} ({count} occurrences)")
        
        print("- Frequent Antecedents:")
        for ant, count in cluster["top_antecedents"]:
            print(f"  - {ant} ({count} occurrences)")
        
        print("- Top Mechanism Examples:")
        for j, mech in enumerate(cluster["top_mechanisms"][:3]):
            print(f"  {j+1}. {mech}")
        
        print("-" * 60)

# Execute analysis
if __name__ == "__main__":
    # Configure paths
    data_path = r"./encode_result.txt"
    json_output_path = r"./mechanism_clusters.json"
    report_output_path = r"./analysis_report.txt"
    
    # Load data
    logger.info(f"Loading data from: {data_path}")
    cam_data = load_cam_data(data_path)
    
    if not cam_data:
        logger.error("Error: No CAM data loaded, check file path and format")
    else:
        # Perform clustering analysis
        logger.info("Starting clustering analysis...")
        mechanism_clusters = cluster_mechanisms(
            cam_data, 
            similarity_threshold=0.70,
            top_k=8
        )
        
        # Save JSON results
        save_json_results(mechanism_clusters, json_output_path)
        
        # Save text report
        save_text_report(mechanism_clusters, report_output_path, top_n=10)
        
        # Print report to console
        print_analysis_report(mechanism_clusters, top_n=10)
